"""
Burst dictionary creation and dimensionality reduction of
burst waveforms.
"""

import numpy as np
from os.path import join

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from mne.decoding import CSP as CSP
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt

from help_funcs import check_existence, ascertain_trials, dict_bycondition
from plot_burst_dict import (
    plot_burst_characteristics,
    plot_burst_chars_conditioned,
    plot_drm_waveforms,
    plot_score_modulation,
    plot_waveforms_score,
)


class BurstSpace:
    """
    Creation of a dictionary containing all detected bursts from a
    single or an arbritrary number of subjects.

    Parameters
    ----------
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    subjects: int or list
              Index or list containing the indices of the subjects to
              be analyzed.
    channel_ids: list or str, optional
                 Indices of channels to take into account during burst
                 dictionary creation. If set to "all" take into account
                 all available channels.
                 Defaults to "all".
    trials_fraction: float, optional
                     Fraction of total trials in each subject's data
                     used to create the burst dictionary.
                     Defaults to 0.2.
    tf_method: str {"wavelets", "superlets", "lagged_coherence"}, optional
               String indicating the algorithm used for burst
               extraction.
               Defaults to "superlets".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    band: str {"mu", "beta"}, optional
          Select band for burst detection.
          Defaults to "beta".
    stratification: int, str or None, optional
                    If set to 'int', split the dictionary's trials in (almost)
                    equal strata; the assumption is that you are interested in
                    a repeated estimation of the dictionary and the dimensionality
                    reduction model. If 'str', try to match it to an experimental
                    session. If None use a random sample of all trials according
                    to "trials_fraction".
                    Defaults to "None".
    strata_id: int or None, optional
               If set to 'int', used as an index of the strata. Ignored if
               'stratification' is set to "None".
               Defaults to "None".
    threshold_feature: str or None {"amplitude", "volume", "duration", "cycles",
                                   "fr_span", "peak_time", "peak_fr", None}, optional
                       If no set to "None" only use bursts with a feature
                       above a certain percentile. See the '_feature_threshold'.
                       Defaults to None.
    percentile: int, optional
                Only used if 'self.threshold_feature' is set to True.
                Percentile of amplitude distribution for bursts in the dictionary.
                Any burst below this percentile will not be considered.
                Defaults to 75.
    verbose: bool, optional
             Control the verbosity of the algorithm.
             Defaults to "True".


    Attributes
    ----------
    channels: list
              Names of available channels.
    tmin, tmax: float
                Start and end time of the epochs in seconds, relative to
                the time-locked event.
    sfreq: int
           Sampling frequency of the recordings in Hz.
    exp_time: 1D numpy array
              Experimental (cropped) time axis.
    exp_time_periods: 4-element list or 1D array
                      Beginning of baseline period, beginning of task period,
                      end of task period, and end of rebound period (in seconds
                      relative to the time-locked event).
    bin_dt: float
            Time window length for binning the time axis.
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    fooof_save_str: str
                    String that loads the appropriate burst data, depending
                    on the 'remove_fooof' parameter.
    """

    def __init__(
        self,
        exp_variables,
        subjects,
        channel_ids="all",
        trials_fraction=0.2,
        tf_method="superlets",
        remove_fooof=True,
        band="beta",
        stratification=None,
        strata_id=None,
        threshold_feature=None,
        percentile=75,
        verbose=True,
    ):
        self.exp_variables = exp_variables
        self.subjects = subjects
        self.channel_ids = channel_ids
        self.trials_fraction = trials_fraction
        self.tf_method = tf_method
        self.remove_fooof = remove_fooof
        self.band = band
        self.stratification = stratification
        self.strata_id = strata_id
        self.threshold_feature = threshold_feature
        self.percentile = percentile
        self.verbose = verbose

        # Direct access to variables, and time axis creation.
        if exp_variables["dataset"] == "munichmi":
            self.channels = exp_variables["_channels"]
        else:
            self.channels = exp_variables["channels"]

        self.exp_time_periods = exp_variables["exp_time_periods"]
        self.savepath = exp_variables["dataset_path"]
        self.bin_dt = exp_variables["bin_dt"]

        self.tmin = exp_variables["tmin"]
        self.tmax = exp_variables["tmax"]
        self.sfreq = exp_variables["sfreq"]
        self.exp_time = np.linspace(
            self.tmin,
            self.tmax,
            int((np.abs(self.tmax - self.tmin)) * self.sfreq) + 1,
        )
        self.exp_time = np.around(self.exp_time, decimals=3)

        # String for correct burst data retrieval based on aperiodic fit
        # subtraction choice.
        if self.remove_fooof == True:
            self.fooof_save_str = ""
        elif self.remove_fooof == False:
            self.fooof_save_str = "_nfs"

    def _load_bursts(
        self, first_subject, burst_data, burst_dict, common_trials, sample_trials
    ):
        """
        Parse all detected bursts of a single subject and append them in
        the burst dictionary.

        Parameters
        ----------
        first_subject: bool
                       Flag indicating the index of a subject in a list.
                       Useful for handling differently the case of the 1st
                       subject (insert data) versus the rest (append data).
        burst_data: numpy array
                    Array of dictionaries, each one containing the detected
                    bursts corresponding to a signle channel.
        burst_dict: dict
                    Dictionary to be filled with all detected bursts.
        common_trials: list or numpy array
                       List containing all trilas with detected bursts across
                       channels.
        sample_trials: numpy array
                       Array containing the the trials on which the
                       burst dictionary creation will be based.
        """

        # Total number of channels for single subject.
        unique_channels = []
        for d in range(len(burst_data)):
            try:
                unique_channels.append(np.unique(burst_data[d]["channel"])[0])
            except:
                continue

        first_channel = True
        for ch in unique_channels:
            # Potentially skip channels of no interest.
            if self.channel_ids != "all" and ch not in self.channel_ids:
                continue

            # Potentially skip channels without any detected bursts and inform.
            # (edge case where periodic activity exists, but not bursts have been
            # detected)
            if len(burst_data[ch]["trial"]) == 0:
                if self.verbose == True:
                    print("\tNo bursts found in channel {}!.".format(ch + 1))
                continue

            # Select common trials.
            common_ids = [
                np.where(burst_data[ch]["trial"] == ct)[0] for ct in common_trials
            ]
            common_ids = np.hstack(common_ids)

            # Use all trials.
            if len(sample_trials) == 0:
                # Store all bursts in a common dictionary.
                for key, value in burst_dict.items():
                    # Insert values in all keys the 1st time.
                    if first_subject == True and first_channel == True:
                        if key != "waveform_times":
                            burst_dict[key] = burst_data[ch][key][common_ids]
                        else:
                            burst_dict[key] = burst_data[ch][key]
                    # Following, append values to some of the keys.
                    else:
                        if key == "waveform_times":
                            continue
                        elif key == "waveform":
                            burst_dict[key] = np.vstack(
                                (value, burst_data[ch][key][common_ids])
                            )
                        else:
                            burst_dict[key] = np.hstack(
                                (value, burst_data[ch][key][common_ids])
                            )

                first_channel = False

            # Use a sample of the total trials.
            elif len(sample_trials) > 0:
                # Find the corresponding trials.
                t_bursts = []
                t_bursts.append(
                    [
                        np.where(burst_data[ch]["trial"][common_ids] == st)[0]
                        for st in sample_trials
                    ]
                )

                for i in range(len(sample_trials)):
                    # Store all bursts in a common dictionary.
                    for key, value in burst_dict.items():
                        # Insert values in all keys the 1st time.
                        if first_subject == True and first_channel == True and i == 0:
                            if key != "waveform_times":
                                burst_dict[key] = burst_data[ch][key][common_ids][
                                    t_bursts[0][i]
                                ]
                            else:
                                burst_dict[key] = burst_data[ch][key]
                        # Following, append values to some of the keys.
                        else:
                            if key == "waveform_times":
                                continue
                            elif key == "waveform":
                                burst_dict[key] = np.vstack(
                                    (
                                        value,
                                        burst_data[ch][key][common_ids][t_bursts[0][i]],
                                    )
                                )
                            else:
                                burst_dict[key] = np.hstack(
                                    (
                                        value,
                                        burst_data[ch][key][common_ids][t_bursts[0][i]],
                                    )
                                )

                first_channel = False

    def _feature_threshold(self):
        """
        Restriction of burst dictionary to a selection of bursts above a
        threshold for a specific feature of the dictionary.

        Attrubutes
        -------
        burst_dictionary: dict
                          Dictionary containing all detected bursts.
        threshold_feature: str or None {"amplitude", "volume", "duration", "cycles",
                                       "fr_span", "peak_time", "peak_fr", None}, optional
                           If no set to "None" only use bursts with a feature
                           above a certain percentile. See the '_feature_threshold'.
                           Defaults to None.
        percentile: int, optional
                    Percentile of feature distribution for bursts in the dictionary.
                    Any burst below this percentile will not be considered.
                    Defaults to 75.
        """

        # Feature selection.
        if self.threshold_feature == "amplitude":
            feat_str = "peak_amp_iter"
        elif self.threshold_feature == "volume":
            feat_str = "volume"
        elif self.threshold_feature == "duration":
            feat_str = "fwhm_time"
        elif self.threshold_feature == "cycles":
            feat_str = "cycles"
        elif self.threshold_feature == "fr_span":
            feat_str = "fwhm_freq"
        elif self.threshold_feature == "peak_time":
            feat_str = "peak_time"
        elif self.threshold_feature == "peak_fr":
            feat_str = "peak_freq"

        # Threshold.
        thr = np.where(
            self.burst_dictionary[feat_str]
            > np.percentile(self.burst_dictionary[feat_str], self.percentile)
        )[0]

        for key, value in self.burst_dictionary.items():
            if key != "waveform_times":
                self.burst_dictionary[key] = value[thr]
            else:
                self.burst_dictionary[key] = value

    def _create(self):
        """
        Creation of a dictionary containing all detected bursts from a
        single or an arbritrary number of subjects. All detected bursts
        are re-arranged in a single dictionary.

        The function looks for bursts detected using different options
        according to the 'TfBursts' class of 'burst_analysis.py' file.

        Optionally subsample the dictionary based on the distribution
        of a single feature. See the '_feature_threshold' function.

        Optionally, summary figures are automatically generated and saved.

        Attrubutes
        -------
        burst_dictionary: dict
                          Dictionary containing all detected bursts.
        drm_trials: list
                    List containing the indices of the randomly drawn trials
                    for creating the dictionary (sample size based on the
                    'self.trials_fraction' or 'self.stratification' attribute).
        missing_trials: list
                        List of trials which are not considered during the
                        dictionary creation due to luck of detected bursts
                        in one or more channels.
        threshold_feature: str or None {"amplitude", "volume", "duration", "cycles",
                                       "fr_span", "peak_time", "peak_fr", None}, optional
                           If not set to "None" only use bursts with a feature
                           above a certain percentile. See the '_feature_threshold'.
                           Defaults to None.
        """

        # Dictionary to store all data.
        self.burst_dictionary = {
            "subject": np.array([]),
            "channel": np.array([]),
            "trial": np.array([]),
            "label": np.array([]),
            "waveform": np.array([]),
            "peak_freq": np.array([]),
            "peak_amp_iter": np.array([]),
            "peak_amp_base": np.array([]),
            "peak_time": np.array([]),
            "peak_adjustment": np.array([]),
            "fwhm_freq": np.array([]),
            "fwhm_time": np.array([]),
            "polarity": np.array([]),
            "volume": np.array([]),
            "waveform_times": np.array([]),
        }

        # Initialization of list that stores the trials used while creating
        # the burst dictionary.
        self.drm_trials = []

        # Initialization of list that stores skipped trials due to luck of
        # bursts in either or both channels.
        self.missing_trials = []

        # If looking into data from single subject convert to list first.
        subjects = self.subjects
        if isinstance(self.subjects, int):
            subjects = [self.subjects]

        first_subject = True
        for subject in subjects:
            # Data loading.
            if self.verbose == True:
                print("Loading bursts of subject {}...".format(subject))
            sub_dir = join(self.savepath, "sub_{}/".format(subject))
            bursts_filename = join(
                sub_dir,
                "{}_bursts_{}{}.npy".format(
                    self.band, self.tf_method, self.fooof_save_str
                ),
            )
            check_existence(bursts_filename)
            sub_bursts = np.load(bursts_filename, allow_pickle=True)

            # Guarantee same trials across channels.
            if self.verbose == True:
                print("Loading corresponding trials' labels...")
            labels = np.load(join(self.savepath, "sub_{}/labels.npy".format(subject)))
            common_trials, missing_trials = ascertain_trials(
                sub_bursts, self.channel_ids, labels
            )
            n_trials = len(common_trials[0]) + len(common_trials[1])
            h_trials = [len(common_trials[0]), len(common_trials[1])]
            self.missing_trials.append(missing_trials)

            # Trial selection.
            if isinstance(self.stratification, str):
                # Choose all trials from corresponding experimental session.
                meta = np.load(
                    join(self.savepath, "sub_{}/meta.npy".format(subject)),
                    allow_pickle=True,
                )

                # Sample ids and trials are the same, as all trials of a
                # session are used for the dictionary.
                sample_ids = np.where(meta[:, 1] == self.stratification)[0]
                
                if len(sample_ids) == 0:
                    msg = "You have specified session '{}' for burst sampling, but it does not match any available experimental session.".format(
                        self.stratification
                    )
                    raise ValueError(msg)

                sample_trials = np.sort(
                    np.hstack(
                        [
                            np.array(common_trials[0]),
                            np.array(common_trials[1]),
                        ]
                    )
                )[sample_ids]

                self.drm_trials.append(sample_ids)

            elif self.stratification == None:
                if self.trials_fraction > 0.0 and self.trials_fraction < 1.0:
                    # Randomly drawn trials for dictionary creation encompassing
                    # both labels.
                    size_0 = int(self.trials_fraction * h_trials[0])
                    size_1 = int(self.trials_fraction * h_trials[1])
                    sample_ids = [
                        (
                            np.random.default_rng().choice(
                                h_trials[0],
                                size=size_0 if size_0 != 0 else 1,
                                replace=False,
                            )
                        ),
                        (
                            np.random.default_rng().choice(
                                h_trials[1],
                                size=size_1 if size_1 !=0 else 1,
                                replace=False,
                            )
                        ),
                    ]

                    # Keep track of the trials used during dictionary creation.
                    temp = np.array(
                        list(set.union(*map(set, [c for c in common_trials])))
                    )
                    sample_trials = np.hstack(
                        [
                            np.array(common_trials[0])[sample_ids[0]],
                            np.array(common_trials[1])[sample_ids[1]],
                        ]
                    )

                    common_sample_ids = []
                    for j in sample_trials:
                        common_sample_ids.append(np.where(temp == j)[0])
                    try:
                        common_sample_ids = list(
                            set.union(*map(set, [c for c in common_sample_ids]))
                        )
                    except:
                        if self.verbose == True:
                            print("Subject with no drm trials: {}".format(subject))
                    self.drm_trials.append(np.array(common_sample_ids))

                elif self.trials_fraction == 1.0:
                    # Use all trials.
                    sample_trials = []

                elif self.trials_fraction <= 0.0 or self.trials_fraction > 1.0:
                    raise ValueError(
                        "The fraction of trials can only be a float > 0 and <= 1."
                    )

            elif self.stratification == 0 or self.stratification > n_trials:
                msg = (
                    "The number of strata must be a positive integer, not greater "
                    + "than the numnber of total available trials: {}.".format(n_trials)
                )
                raise ValueError(msg)

            else:
                # Shuffle the order of the trials pseudo-randomly.
                prng = np.random.RandomState(123467890)
                sfl_trials = np.copy(common_trials)
                sfl_ids = [np.arange(0, h_trials[0], 1), np.arange(0, h_trials[1], 1)]
                prng.shuffle(sfl_ids[0])
                prng.shuffle(sfl_ids[1])

                # Select all trials belonging to a stratum.
                strata_trials = [
                    int(h_trials[0] / self.stratification),
                    int(h_trials[1] / self.stratification),
                ]

                sample_trials = []
                sample_ids = []
                for st, sflt, sfli in zip(strata_trials, sfl_trials, sfl_ids):
                    # Keep trials of stratum.
                    if self.strata_id < self.stratification - 1:
                        start = self.strata_id * st
                        end = start + st
                        batch = np.array(sflt)[sfli][start:end]
                        sample_trials.append(batch)
                    else:
                        start = self.strata_id * st
                        batch = np.array(sflt)[sfli][start:]
                        sample_trials.append(batch)

                    # Find ids of stratum with respect to original trial order.
                    cs_ids = []
                    for b in batch:
                        cs_ids.append(
                            int(np.where(np.sort(np.hstack(common_trials)) == b)[0])
                        )
                    sample_ids.append(cs_ids)

                # Keep track of the trials used during dictionary creation.
                sample_trials = np.hstack(sample_trials)
                sample_ids = np.hstack(sample_ids)
                self.drm_trials.append(sample_ids)

            # Dictionary creation.
            common_trials = np.hstack(common_trials)
            self._load_bursts(
                first_subject,
                sub_bursts,
                self.burst_dictionary,
                common_trials,
                sample_trials,
            )

            first_subject = False

        # Expansion of dictionary with "cycles" information.
        self.burst_dictionary["cycles"] = (
            self.burst_dictionary["fwhm_time"] * self.burst_dictionary["peak_freq"]
        )

        # Optional threshold with respect to a specific feature.
        if self.threshold_feature != None:
            self._feature_threshold()

        # Inform about the number of used bursts.
        if self.verbose == True:
            print(
                "Total number of bursts in dictionary: {}.".format(
                    len(self.burst_dictionary["trial"])
                )
            )

    def compute_feature_modulation(
        self, compute_feature="rate", rate_normalization=True, return_average=False
    ):
        """
        Computation of the modulation of a specific feature per channel and trial
        for the all the bursts of the dictionary.

        Parameters
        ----------
        compute_feature: str {"rate", "amplitude", "volume", "duration", "fr_span",
                             "peak_fr", "cycles"}, optional
                        Feature to examine. "Rate" computes the burst count or rate,
                        depending on the 'rate_normalization' parameter; "amplitude"
                        and "volume" compute the corresponding sum of values; "duration",
                        "fr_span", "peak_fr" and "cycles" compute the corresponding average
                        value.
                        Defaults to "rate".
        rate_normalization: bool, optional
                            If set to "True", normalize count of bursts to burst
                            rate.
                            Defaults to "False".
        return_average: bool, optional
                        If set to "True" return the average value during the
                        task period.
                        Defaults to "False".

        Returns
        -------
        burst_feature: np.array
                       2D or 3D array containing the modulation of the selected feature.
        """

        # Dictionary creation.
        self._create()

        # Feature selection.
        if compute_feature == "amplitude":
            feat_str = "peak_amp_iter"
        elif compute_feature == "volume":
            feat_str = "volume"
        elif compute_feature == "duration":
            feat_str = "fwhm_time"
        elif compute_feature == "fr_span":
            feat_str = "fwhm_freq"
        elif compute_feature == "peak_fr":
            feat_str = "peak_freq"

        # Initialize array [#trials, #channels, binned_time]
        trials = np.unique(self.burst_dictionary["trial"])
        channels = np.unique(self.burst_dictionary["channel"])

        # Binned time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
            baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        erds_time_lims = [baseline_begin, rebound_end]

        erds_time = self.exp_time[erds_time_lims[0] : erds_time_lims[1] + 1]

        binning = np.arange(erds_time[0], erds_time[-1] + self.bin_dt, self.bin_dt)
        binned_erds_time = np.around(binning, decimals=2)

        burst_feature = np.zeros(
            (len(trials), len(channels), len(binned_erds_time) - 1)
        )

        # For each trial...
        for tr, trial in enumerate(trials):
            # For each channel...
            for ch, channel in enumerate(channels):
                # Burst count or rate.
                if compute_feature == "rate":
                    ids = np.where(
                        (self.burst_dictionary["trial"] == trial)
                        & (self.burst_dictionary["channel"] == channel)
                    )[0]

                    burst_feature[tr, ch, :], _ = np.histogram(
                        self.burst_dictionary["peak_time"][ids], bins=binned_erds_time
                    )

                    # Optionally convert burst count to rate.
                    if rate_normalization == True:
                        burst_feature[tr, ch, :] = (
                            burst_feature[tr, ch, :] / self.bin_dt
                        )

                # Rest of features in time-resolved manner.
                else:
                    for t_idx in range(len(binned_erds_time) - 1):
                        # Bursts within a specific time window.
                        ids = np.where(
                            (self.burst_dictionary["trial"] == trial)
                            & (self.burst_dictionary["channel"] == channel)
                            & (
                                self.burst_dictionary["peak_time"]
                                >= binned_erds_time[t_idx]
                            )
                            & (
                                self.burst_dictionary["peak_time"]
                                < binned_erds_time[t_idx + 1]
                            )
                        )[0]

                        if (
                            compute_feature == "amplitude"
                            or compute_feature == "volume"
                        ):
                            burst_feature[tr, ch, t_idx] = np.sum(
                                self.burst_dictionary[feat_str][ids]
                            )
                        elif (
                            compute_feature == "duration"
                            or compute_feature == "fr_span"
                            or compute_feature == "peak_fr"
                        ):
                            burst_feature[tr, ch, t_idx] = np.mean(
                                self.burst_dictionary[feat_str][ids]
                            )

        # Optionally return the average within the task period.
        if return_average == True:
            try:
                task_begin = int(
                    np.where(binned_erds_time == self.exp_time_periods[1])[0]
                )
                task_end = int(
                    np.where(binned_erds_time == self.exp_time_periods[2])[0]
                )
            except:
                task_begin = int(
                    np.where(
                        binned_erds_time == self.exp_time_periods[1] + self.bin_dt / 2
                    )[0]
                )
                task_end = int(
                    np.where(
                        binned_erds_time == self.exp_time_periods[2] + self.bin_dt / 2
                    )[0]
                )
            burst_feature = np.nanmean(
                burst_feature[:, :, task_begin : task_end + 1], axis=-1
            )

        return burst_feature

    def compute_chars_dists(self, subject, winsorization=[2, 98]):
        """
        Compute the distribution of specific burst characteristics' values
        for a given subject.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects' data that are projected to
                 the across-subjects burst dictionary.
        winsorization: None or two-elements list, optional
                       Option to clip bursts with exreme values along a given
                       burst characteristic to certain limits of the
                       interquantile range.
                       Defaults to [2, 98].
        """

        # Dictionary to store all data.
        subject_dictionary = {
            "subject": np.array([]),
            "channel": np.array([]),
            "trial": np.array([]),
            "label": np.array([]),
            "waveform": np.array([]),
            "peak_freq": np.array([]),
            "peak_amp_iter": np.array([]),
            "peak_amp_base": np.array([]),
            "peak_time": np.array([]),
            "peak_adjustment": np.array([]),
            "fwhm_freq": np.array([]),
            "fwhm_time": np.array([]),
            "polarity": np.array([]),
            "volume": np.array([]),
            "waveform_times": np.array([]),
        }

        # Dictionary creation.
        self._create()

        # Individualized bursts.
        sub_bursts = np.where(self.burst_dictionary["subject"] == subject)[0]
        for key, value in self.burst_dictionary.items():
            # Insert values in all keys the 1st time.
            if key != "waveform_times":
                subject_dictionary[key] = value[sub_bursts]
            else:
                subject_dictionary[key] = value

        # Burst characteristics.
        chars_dists = []
        chars = ["volume", "peak_amp_iter", "peak_freq", "fwhm_time", "fwhm_freq"]
        for char in chars:
            char_dist = subject_dictionary[char]

            # Optional winsorization.
            if winsorization != None:
                cut = np.array(
                    [
                        np.percentile(char_dist, winsorization[0]),
                        np.percentile(char_dist, winsorization[1]),
                    ]
                )
                char_dist[char_dist < cut[0]] = cut[0]
                char_dist[char_dist > cut[1]] = cut[1]

            chars_dists.append(char_dist)

        return subject_dictionary, chars_dists

    def _apply_solver(self, burst_waveforms, n_components=0.9):
        """
        Adjustment of dimensionality reduction step based on the selected algorithm.
        The function instantiates a class model based on the 'fit_transform' function
        of the selected 'solver', and the corresponding components.

        The CSP algorithm optimizes the decomposition based on the laterality of
        the recordings compared to the condition (aka movement labels).

        Parameters
        ----------
        burst_waveforms: numpy array
                         Numpy array containing the waveforms of all bursts that
                         will be used to fit the dimensionality reduction model.
                         In the case of PCA, it corresponds to a 2D matrix of
                         dimensions [#bursts, #waveform_time_points]. In the case
                         of CSP it corresponds to a 3D tensor of dimensions
                         [#bursts, #waveform_time_points, 1], the last dimension
                         being null and existing only bacause it is expected by
                         the MNE-python CSP class.
        n_components: int or float
                      Number of components to be returned by the scikit PCA function
                      for any dimensionality reduction method, or percentage of
                      explained variance if 'solver' is set to "pca".
                      Defaults to 0.90, assuming that 'solver' is set to the
                      default "pca".

        Attributes
        ----------
        drm: scikit-learn or MNE-python model
             Fitted-transformed model returned from the corresponding sklearn, or
             MNE-python function.
        components: numpy array
                    Array containing the transformed bursts as returned by the
                   'fit_transform' method of PCA or CSP.
        """

        # Abort if n_components is fraction, but the solver does not correspond to PCA.
        if isinstance(n_components, float) and self.solver != "pca":
            warn = (
                "When not using PCA as a dimensionality reduction technique, "
                + "the number of components should be an integer, not float."
            )
            raise ValueError(warn)

        # Fit transform.
        if self.solver == "pca":
            self.drm = PCA(n_components=n_components, svd_solver="full")
            self.components = self.drm.fit_transform(burst_waveforms)

        elif self.solver == "csp":
            # Create labels for CSP algorithm.
            unique_labels = np.unique(self.burst_dictionary["label"])

            # HARD CODED ASSUMPTION: C3 COMES BEFORE C4, BECAUSE THE LABELS ARE ORDERED
            # ALPHABETICALLY.
            c3_ipsi = np.where(
                (self.burst_dictionary["channel"][self.drm_ids] == self.channel_ids[0])
                & (self.burst_dictionary["label"][self.drm_ids] == unique_labels[0])
            )[0]
            c3_contr = np.where(
                (self.burst_dictionary["channel"][self.drm_ids] == self.channel_ids[0])
                & (self.burst_dictionary["label"][self.drm_ids] == unique_labels[1])
            )[0]
            c4_ipsi = np.where(
                (self.burst_dictionary["channel"][self.drm_ids] == self.channel_ids[1])
                & (self.burst_dictionary["label"][self.drm_ids] == unique_labels[1])
            )[0]
            c4_contr = np.where(
                (self.burst_dictionary["channel"][self.drm_ids] == self.channel_ids[1])
                & (self.burst_dictionary["label"][self.drm_ids] == unique_labels[0])
            )[0]

            labels = self.burst_dictionary["label"][self.drm_ids]
            labels[c3_ipsi] = "ipsi"
            labels[c4_ipsi] = "ipsi"
            labels[c3_contr] = "contr"
            labels[c4_contr] = "contr"

            burst_waveforms = burst_waveforms.reshape(
                burst_waveforms.shape[0], burst_waveforms.shape[1], 1
            )

            self.drm = CSP(
                n_components=n_components,
                reg=None,
                log=None,
                transform_into="csp_space",
                component_order="alternate",
            )
            self.components = self.drm.fit_transform(burst_waveforms, labels)[:, :, 0]

    def fit_transform(self, solver="pca", n_components=0.90, output="cv_burst_rate"):
        """
        Projection of all the bursts present in the burst dictionary to an alternative
        space (dimensionality reduction model).

        Each subject's bursts are first z-scored in order to ensure the analysis uses
        the same numerical range.

        Optionally plot summary figures.

        Parameters
        ----------
        subjects: int or list
                  Index or list containing the indices of the subjects to
                  be analyzed.
        solver: str {"pca", "csp"}, optional
                Dimensionality reduction algorithm. Implements the PCA sklearn model,
                or the MNE-python CSP model.
                Defaults to "pca".
        n_components: int or float, optional
                      Number of principal components or percentage of explained
                      variance to be returned by the scikit PCA function.
                      Defaults to 0.90, assuming that 'solver' is set to the
                      default "pca".
        output: str {"cv_burst_rate", "waveforms", "plots"}, optional
                If set to "cv_burst_rate" use only 'drm_ids' during model fitting,
                in order to preserve a validation trial set. Else use all
                drm trials.
                Defaults to "cv_burst_rate".

        Attributes
        ----------
        solver: str {"pca", "csp"}, optional
                Dimensionality reduction algorithm. Implements the PCA sklearn model,
                or the MNE-python CSP model.
                Defaults to "pca".
        val_ids: list
                 List of lists, each one corresponding to the indices of drm_trials
                 than are reserved for validation of classification results per
                 subject and stratum.
                 Only
        drm_ids: list
                 List containing the indices of drm_trials than are reserved for
                 used during the dimenionality reduction model fitting.
        all_subs_bursts: np.array
                         Array containing all bursts used for fitting the model
                         after robust scaling.
        """

        # Dictionary creation.
        self.solver = solver
        self._create()

        # If looking into data from single subject convert to list first.
        subjects = self.subjects
        if isinstance(self.subjects, int):
            subjects = [self.subjects]

        # Initialization of list that stores the validation trials for
        # the burst dictionary.
        self.val_ids = []
        self.drm_ids = []

        # Space transformation.
        all_subs_bursts = []
        for sub in subjects:
            # Subject-specific bursts.
            sub_bursts = np.where(self.burst_dictionary["subject"] == sub)[0]

            # Individualized bursts scaling.
            scaler = RobustScaler()

            # Burst dictionary for waveform-resolved burst rate computation
            # requires a cross-validation set.
            if output == "cv_burst_rate":
                # Separation of bursts for dimensionality reduction technique
                # and validation set.
                labels = np.load(join(self.savepath, "sub_{}/labels.npy".format(sub)))
                labels_trials = self.burst_dictionary["trial"][sub_bursts]
                labels_trials_unique = np.unique(labels_trials)
                model_labels = labels[np.unique(labels_trials)]

                labels_1 = np.where(model_labels == np.unique(labels)[0])[0]
                labels_2 = np.where(model_labels == np.unique(labels)[1])[0]

                # "Balanced" labels.
                if len(labels_1) >= 2 and len(labels_2) >= 2:
                    drm_trials_real = np.hstack([labels_1[0::2], labels_2[1::2]])
                    self.val_ids.append(np.hstack([labels_1[1::2], labels_2[0::2]]))
                elif len(labels_1) == 1 and len(labels_2) >= 2:
                    drm_trials_real = labels_2[1::2]
                    self.val_ids.append(np.hstack([labels_1, labels_2[0::2]]))
                elif len(labels_1) >= 2 and len(labels_2) == 1:
                    drm_trials_real = labels_1[0::2]
                    self.val_ids.append(np.hstack([labels_1[1::2], labels_2]))

                drm_ids = []
                for j in drm_trials_real:
                    drm_ids.append(
                        np.where(labels_trials == labels_trials_unique[j])[0]
                    )
                drm_ids = list(set.union(*map(set, [c for c in drm_ids])))
                self.drm_ids.append(drm_ids)

                # Scaling model fitting.
                standardized_bursts = scaler.fit_transform(
                    self.burst_dictionary["waveform"][drm_ids]
                )

            # Waveforms computation and plotting can use all data.
            elif output == "waveforms" or output == "plots":
                # Scaling model fitting.
                standardized_bursts = scaler.fit_transform(
                    self.burst_dictionary["waveform"][sub_bursts]
                )

            all_subs_bursts.append(standardized_bursts)

        # Aggregate bursts from all subjects.
        all_subs_bursts = np.vstack(all_subs_bursts)
        self.all_subs_bursts = all_subs_bursts

        # Aggregate trial indices that to be used in dimensionality
        # reduction model fitting.
        if output == "cross_validation":
            self.drm_ids = np.hstack(self.drm_ids)

        self._apply_solver(all_subs_bursts, n_components=n_components)

    def _dist_scores(
        self, comps_to_analyze, bursts_projection, winsorization=None, drm_dist=False
    ):
        """
        Compute the distribution of scores in a given subset a trials.

        If winsorization is applied to waveforms of trials used in order
        to create the dimensionality reduction model, then the same extreme
        values are used in order to bound the scores for all trials.

        Parameters
        ----------
        comps_to_analyze: list or numpy array
                          List of the indices of components used for
                          feature creation.
        bursts_projection: numpy array
                           Array of transformed burst waveforms.
        winsorization: None or two-elements list, optional
                       Option to clip bursts with exreme values along a given
                       component axis to certain limits of the interquantile range.
                       Defaults to "None".
        drm_dist: bool, optional
                  Flag indicating whether currently processing bursts from trials
                  used while creating the dimensionality reduction model or not.
                  Defualts to "False".

        Returns
        -------
        scores_dists: list
                      List containing all burst scores along each principal
                      component axis.
        win_limits: list
                    List containing burst scores along each principal
                    component axis that correspond to the winsorization
                    limits.
        """

        # Feature creation.
        scores_dists = []
        win_limits = []
        for pc_id, comp in enumerate(comps_to_analyze):
            scores_dist = bursts_projection[:, comp - 1]

            # Optional winsorization will clip outlier values to
            # the 2, 98 IQRs, based on the trials that were used
            # while constructing the dimensionality reduction model.
            if winsorization != None and drm_dist == True:
                cut = np.array(
                    [
                        np.percentile(scores_dist, winsorization[0]),
                        np.percentile(scores_dist, winsorization[1]),
                    ]
                )
                scores_dist[scores_dist < cut[0]] = cut[0]
                scores_dist[scores_dist > cut[1]] = cut[1]

                win_limits.append(cut)

            elif winsorization != None and drm_dist == False:
                cut = self.drm_win_limits[pc_id]
                scores_dist[scores_dist < cut[0]] = cut[0]
                scores_dist[scores_dist > cut[1]] = cut[1]

                win_limits.append(cut)

            scores_dists.append(scores_dist)

        return scores_dists, win_limits

    def transform_sub(self, subject, comps_to_analyze, winsorization=[2, 98]):
        """
        Burst dictionary creation and projection of all bursts in an already fitted
        dimensionality reduction model.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects' data that are projected to
                 the across-subjects burst dictionary.
        comps_to_analyze: list or numpy array
                          List of the indices of components used for
                          feature creation.
        winsorization: None or two-elements list, optional
                       Option to clip bursts with exreme values along a given
                       component axis to certain limits of the interquantile range.
                       Defaults to [2, 98].

        Attributes
        ----------
        subject_dictionary: dict
                            Dictionary containing all detected bursts of single
                            subject.
        sub_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on all trials).
        drm_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on trials used while creating across-
                          subjects model).
        drm_win_limits: list
                        List containing burst scores along each component axis
                        that correspond to the winsorization limits (based on
                        trials used while creating across-subjects model).

        Returns
        -------
        subject_dictionary: dict
                            Dictionary containing all detected bursts of single
                            subject.
        sub_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on all trials).
        drm_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on trials used while creating
                          across-subjects model).
        """

        # Dictionary to store all data.
        subject_dictionary = {
            "subject": np.array([]),
            "channel": np.array([]),
            "trial": np.array([]),
            "label": np.array([]),
            "waveform": np.array([]),
            "peak_freq": np.array([]),
            "peak_amp_iter": np.array([]),
            "peak_amp_base": np.array([]),
            "peak_time": np.array([]),
            "peak_adjustment": np.array([]),
            "fwhm_freq": np.array([]),
            "fwhm_time": np.array([]),
            "polarity": np.array([]),
            "volume": np.array([]),
            "waveform_times": np.array([]),
        }

        # Ensure that the requested number of components does not exceed the available.
        if (
            self.solver == "pca" and len(comps_to_analyze) > self.drm.n_components_
        ) or (self.solver == "csp" and len(comps_to_analyze) > self.drm.n_components):
            raise ValueError(
                "The requested number of components exceeds the available."
            )

        # Subject data.
        sub_dir = join(self.savepath, "sub_{}/".format(subject))
        bursts_filename = join(
            sub_dir,
            "{}_bursts_{}{}.npy".format(self.band, self.tf_method, self.fooof_save_str),
        )
        check_existence(bursts_filename)
        sub_bursts = np.load(bursts_filename, allow_pickle=True)[()]
        labels = np.load(join(self.savepath, "sub_{}/labels.npy".format(subject)))

        # Guarantee same trials across channels.
        common_trials, _ = ascertain_trials(sub_bursts, self.channel_ids, labels)
        common_trials = np.hstack(common_trials)

        # Projection of individual subject's model trials to the common drm space,
        # and estimation of the distributions of scores along each component axis,
        # as well as the winsorization limits.
        if isinstance(self.subjects, list):
            if len(self.drm_trials) > 0:
                sub_id = int(np.where(np.array(self.subjects) == subject)[0])
                sample_trials = self.drm_trials[sub_id]
            else:
                sample_trials = []
        else:
            if len(self.drm_trials) > 0:
                sample_trials = self.drm_trials[0]
            else:
                sample_trials = []

        first_subject = True
        self._load_bursts(
            first_subject, sub_bursts, subject_dictionary, common_trials, sample_trials
        )

        scaler = RobustScaler()
        standardized_bursts = scaler.fit_transform(subject_dictionary["waveform"])
        trans_waveforms = self.drm.transform(standardized_bursts)
        self.drm_scores_dists, self.drm_win_limits = self._dist_scores(
            comps_to_analyze, trans_waveforms, winsorization, drm_dist=True
        )

        # Projection of all trials of a subject's to the common drm space,
        # and estimation of the distributions of scores along each principal
        # component axis, respecting the above winsorization limits.
        sample_trials = []
        first_subject = True
        self._load_bursts(
            first_subject, sub_bursts, subject_dictionary, common_trials, sample_trials
        )

        scaler = RobustScaler()
        standardized_bursts = scaler.fit_transform(subject_dictionary["waveform"])
        trans_waveforms = self.drm.transform(standardized_bursts)
        self.sub_scores_dists, _ = self._dist_scores(
            comps_to_analyze, trans_waveforms, winsorization, drm_dist=False
        )

        return subject_dictionary, self.sub_scores_dists, self.drm_scores_dists

    def estimate_waveforms(
        self,
        comps_to_analyze,
        comps_groups,
        winsorization=[2, 98],
        output_waveforms="extrema",
        waveform_filters="peaks",
        n_comps=3,
    ):
        """
        Transformation of bursts used in the dimensionality reduction model
        acroos each of its axes, and estimation of the burst waveforms in a
        score-resolved manner.

        Parameters
        ----------
        comps_to_analyze: list or numpy array
                          List of the indices of components used for
                          feature creation.
        comps_groups: int
                      Number of groups the scores of each component
                      axis should be split into.
        winsorization: None or two-elements list, optional
                       Option to clip bursts with exreme values along a given
                       principal component axis to certain limits of the
                       interquantile range.
                       Defaults to [2, 98].
        output_waveforms: str {"all", "mid_extrema", "extrema"}, optional
                          Option that controls the returned output of the function.
                          If set to "all" return all estimated waveforms group per
                          component. If set to "mid_extrema" return the middle waveform
                          (around 0 score, corresponding to the origin of all axes).
                          If set to "extrema" return only the two waveforms corresponding
                          to the negative- and positive-extreme groups.
                          Note that if 'comps_groups' is an even number the two groups
                          around the 0 score zero will be merged into one.
                          Defaults to "extrema".
        waveform_filters: str {"peaks", "fwhm"}, optional
                          String that controls how to estimate a filter bank corresponding
                          to the returned waveforms. If set to "peaks" the filter bank spans
                          the frequency range from the lowest to the highest spectral peak.
                          If set to "fwhm" the filter bank is based on the FWHM of the unique
                          spectral peaks.
                          Defaults to "peaks".
        n_comps: int, optional
                 Number of components ultimately kept among those indicated by 'comps_to_analyze',
                 based on the modulation index. This index indicates the relative difference
                 between ipsilateral and contralateral average waveform modulation during the
                 task period relative to baseline for each component.
                 Defaults to 3.

        Returns
        -------
        drm_components: list
                        List containing the drm waveforms corresponding to each component
                        in 'comps_to_analyze'.
        binned_waveforms: list
                          List containing the average, across-conditions burst waveforms per
                          group of each component.
        waveform_freq_ranges: list of lists
                              List containing the kernels' spectra frequency ranges estimates.
        """

        # Returned waveforms list initialization.
        binned_waveforms = []

        # Returned modulation index initialization.
        modulation_index = []

        # Sanity check that the required number of componets to be returned is at most
        # equal to those analyzed.
        if len(comps_to_analyze) < n_comps:
            n_comps = len(comps_to_analyze)

        # Estimation of score distributions per component.
        self.sub_scores_dists, _ = self._dist_scores(
            comps_to_analyze,
            self.components,
            winsorization,
            drm_dist=True,
        )

        # Binned time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
            baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        erds_time_lims = [baseline_begin, rebound_end]

        erds_time = self.exp_time[erds_time_lims[0] : erds_time_lims[1] + 1]

        binning = np.arange(erds_time[0], erds_time[-1] + self.bin_dt, self.bin_dt)
        binned_erds_time = np.around(binning, decimals=2)

        # Task time limits.
        trial_begin = self.exp_time_periods[1]
        trial_end = self.exp_time_periods[2]

        # Modulation index lateralization ids.
        unique_labels = np.unique(self.burst_dictionary["label"])

        c3c4_ids = [
            int(np.where(np.array(self.channels) == "C3")[0]),
            int(np.where(np.array(self.channels) == "C4")[0]),
        ]

        c3_ipsi = np.where(
            (self.burst_dictionary["channel"] == c3c4_ids[0])
            & (self.burst_dictionary["label"] == unique_labels[0])
        )[0]
        c3_contr = np.where(
            (self.burst_dictionary["channel"] == c3c4_ids[0])
            & (self.burst_dictionary["label"] == unique_labels[1])
        )[0]
        c4_ipsi = np.where(
            (self.burst_dictionary["channel"] == c3c4_ids[1])
            & (self.burst_dictionary["label"] == unique_labels[1])
        )[0]
        c4_contr = np.where(
            (self.burst_dictionary["channel"] == c3c4_ids[1])
            & (self.burst_dictionary["label"] == unique_labels[0])
        )[0]

        # Estimation of waveforms and modulation index per component.
        for scd in self.sub_scores_dists:
            comp_waveforms = []

            # Limits in the metrics that are used to split each image in features.
            if comps_groups == 1:
                raise ValueError(
                    "You need to specify at least 2 groups when creating burst features!"
                )
            else:
                # Common score binning across conditions.
                iqrs = np.linspace(np.min(scd), np.max(scd), comps_groups + 1)

                for i in range(comps_groups):
                    # Group score limits.
                    scores_lims = [iqrs[i], iqrs[i + 1]]

                    # Condition specific waveforms.
                    if i != comps_groups - 1:
                        waveform_ids = np.where(
                            (scd >= scores_lims[0]) & (scd < scores_lims[1])
                        )[0]
                    else:
                        waveform_ids = np.where(scd >= scores_lims[0])[0]
                    waveform = np.mean(
                        self.burst_dictionary["waveform"][waveform_ids], axis=0
                    )

                    comp_waveforms.append(waveform)

            # Optionally keep selected waveforms.
            if output_waveforms == "extrema":
                comp_waveforms = [comp_waveforms[0], comp_waveforms[-1]]
            elif output_waveforms == "mid_extrema":
                if comps_groups % 2 != 0:
                    mid = comps_groups // 2
                    comp_waveforms = [
                        comp_waveforms[0],
                        comp_waveforms[mid],
                        comp_waveforms[-1],
                    ]
                elif comps_groups % 2 == 0:
                    mid_neg = comps_groups // 2 - 1
                    mid_pos = comps_groups // 2
                    mid_waveform = np.mean(
                        np.vstack([comp_waveforms[mid_neg], comp_waveforms[mid_pos]]),
                        axis=0,
                    )
                    comp_waveforms = [
                        comp_waveforms[0],
                        mid_waveform,
                        comp_waveforms[-1],
                    ]

            binned_waveforms.append(comp_waveforms)

            # Modulation index.
            modulation_values = np.zeros((4, 1))
            for l, cs in enumerate([c3_ipsi, c3_contr, c4_ipsi, c4_contr]):
                base_ids = self.burst_dictionary["peak_time"][cs] < trial_begin
                task_ids = (self.burst_dictionary["peak_time"][cs] >= trial_begin) & (
                    self.burst_dictionary["peak_time"][cs] <= trial_end
                )

                cs_values_base = np.mean(scd[cs][base_ids]) if any(base_ids) else 0
                cs_values_task = np.mean(scd[cs][task_ids]) if any(task_ids) else 0
                modulation_values[l, 0] = np.abs(cs_values_task - cs_values_base)

            modulation_index.append(
                np.abs(
                    (modulation_values[0, 0] - modulation_values[3, 0])
                    - (modulation_values[2, 0] - modulation_values[1, 0])
                )
            )

        # Add the corresponding axes to the returned list.
        drm_comps_to_return = (np.array(comps_to_analyze) - 1).tolist()
        if self.solver == "pca":
            drm_components = self.drm.components_[drm_comps_to_return, :]
        elif self.solver == "csp":
            drm_components = self.drm.filters_.T[drm_comps_to_return, :]

        # Descending modulation index order up the number of components
        # selected to be used.
        modulation_index_max = np.argsort(-np.array(modulation_index))[:n_comps]

        # Subselection of components based on modulation index.
        binned_waveforms_temp = []
        for mim in modulation_index_max:
            binned_waveforms_temp.append(binned_waveforms[mim])
        binned_waveforms = binned_waveforms_temp
        drm_components = drm_components[modulation_index_max]

        # Estimation of spectral content of waveforms, based on which a filter
        # bank is created.
        waveform_fft_specta = []
        fft_freqs = rfftfreq(len(binned_waveforms[0][0]), d=1 / self.sfreq)

        for k, comp_waveforms in enumerate(binned_waveforms):
            for l, comp_waveform in enumerate(comp_waveforms):
                fft_ampl = np.abs(rfft(comp_waveform))

                # Peaks and FHWM.
                fft_peaks, _ = find_peaks(fft_ampl, height=np.std(fft_ampl))
                _, _, fft_peaks_ll, fft_peaks_rl = peak_widths(fft_ampl, fft_peaks)

                if waveform_filters == "peaks":
                    for fft_peak in fft_freqs[fft_peaks]:
                        waveform_fft_specta.append(fft_peak)

                elif waveform_filters == "fwhm":
                    for left_freq, right_freq in zip(fft_peaks_ll, fft_peaks_rl):
                        waveform_fft_specta.append(
                            [
                                fft_freqs[int(np.around(left_freq))],
                                fft_freqs[int(np.around(right_freq))],
                            ]
                        )

        # Find frequency ranges of kernels.
        if waveform_filters == "peaks":
            waveform_freq_ranges = []

            low_peak = np.min(waveform_fft_specta)
            high_peak = np.max(waveform_fft_specta)
            filter_ranges = np.arange(np.floor(low_peak) - 1, np.ceil(high_peak) + 4, 4)

            for id, range_low in enumerate(filter_ranges[:-1]):
                waveform_freq_ranges.append([range_low, filter_ranges[id + 1]])

        elif waveform_filters == "fwhm":
            unique_waveform_fft_spectra = np.unique(waveform_fft_specta, axis=0)

            # Remove degenerate peaks/ranges corresponding to a peak without
            # a frequency span at least as large as the frequency resolution
            # of the FFT.
            itd = []
            for id, unique_waveform_fft_freq in enumerate(unique_waveform_fft_spectra):
                if unique_waveform_fft_freq[0] == unique_waveform_fft_freq[1]:
                    itd.append(id)
            waveform_freq_ranges = np.around(
                np.delete(unique_waveform_fft_spectra, itd, axis=0)
            )

        return drm_components, binned_waveforms, waveform_freq_ranges, modulation_index_max

    def plot_dict(
        self,
        savefigs=True,
        plot_format="pdf",
        subsets="all",
        comps_to_visualize=[2, 3, 4],
        bprop = 0.002,
    ):
        """
        Visulizations of burst dictionary.

        Parameters
        ----------
        savefigs: bool, optional
                  If set to "True" the visualizations are automatically
                  saved. If set to "False" they are shown on screen.
                  Defaults to "True".
        plot_format: str {"pdf", "png"}, optional
                     File format. Prefer "pdf" for editing with vector graphics
                     applications, or "png" for less space usage and better
                     integration with presentations. Ignored if 'savefigs'
                     is set to "False".
                     Defaults to "pdf".
        subsets: str {"all", "short", "medium", "long"}, optioanl
                 Str that controls which subset of bursts to be plotted with
                 respect to the number of cycles.
                 Defaults to "all".
        comps_to_visualize: list, optional
                            List of components for detailed visualization of
                            corresponding scores. Must contain 2 or 3 elements.
                            Defaults to [2,3,4]
        bprop: float, optional
               Proportion of bursts to be plotted in first visualization.
               Defaults to 0.002.
        """

        # Burst waveforms and distributions of burst characteristics.
        plot_burst_characteristics(
            self.subjects,
            self.burst_dictionary,
            self.tf_method,
            self.band,
            self.savepath,
            bprop=bprop,
            savefigs=savefigs,
            plot_format=plot_format,
        )

        """
        # Distributions of burst characteristics per channel and condition.
        plot_burst_chars_conditioned(
            self.subjects,
            self.burst_dictionary,
            self.exp_time_periods,
            self.channel_ids,
            self.tf_method,
            self.band,
            self.savepath,
            subsets=subsets,
            savefigs=savefigs,
            plot_format=plot_format,
        )
        """

        # Explained variance, score modulation and motifs.
        # Binned time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
            baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            mi_end = int(np.where(self.exp_time == self.exp_time_periods[2])[0])
        except:
            mi_end = np.where(self.exp_time >= self.exp_time_periods[2])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        base_time_lims = [self.exp_time_periods[0], self.exp_time_periods[1]]
        erds_time_lims = [baseline_begin, rebound_end]

        erds_time = self.exp_time[erds_time_lims[0] : erds_time_lims[1] + 1]

        binning = np.arange(erds_time[0], erds_time[-1] + self.bin_dt, self.bin_dt)
        binned_erds_time = np.around(binning, decimals=2)

        # Plot up to 8 components' exaplined variance, score modulation and motifs.
        if self.components.shape[1] < 8:
            n_comps = self.components.shape[1]
        else:
            n_comps = 8

        plot_drm_waveforms(
            self.burst_dictionary,
            self.drm,
            self.components,
            self.channel_ids,
            self.solver,
            binned_erds_time,
            base_time_lims,
            self.exp_time_periods[2],
            self.tf_method,
            self.band,
            self.savepath,
            comps_to_analyze=n_comps,
            savefigs=savefigs,
            plot_format=plot_format,
        )

        """
        # Score modualtion in 2D or 3D space.
        plot_score_modulation(
            self.burst_dictionary,
            self.components,
            self.channel_ids,
            self.solver,
            binned_erds_time,
            base_time_lims,
            self.exp_time_periods[2],
            self.tf_method,
            self.savepath,
            comps_to_visualize=comps_to_visualize,
            savefigs=savefigs,
            plot_format=plot_format,
        )

        # Burst waveforms in different conditions and time windows.
        plot_waveforms_score(
            self.burst_dictionary,
            self.components,
            self.channel_ids,
            self.solver,
            binned_erds_time,
            base_time_lims,
            mi_end,
            self.bin_dt,
            self.tf_method,
            self.band,
            self.savepath,
            snapshot_window=None,
            comps_to_visualize=comps_to_visualize,
            brate=True,
            savefigs=savefigs,
            plot_format=plot_format,
        )
        """
