"""
Feature creation based on burst rate modulation for bursts
with distinct shapes.
"""

import numpy as np

from help_funcs import dict_bycondition
from time_res_features import significant_components
from plot_burst_features import features_modulation, features_modulation_nd
from plot_tf_features import chars_modulation


class BurstFeatures:
    """
    Create features of burst rate modulation in time, suitable for binary
    classification tasks.

    Given the burst activity of a single subject, project all bursts in a
    predefined dimensionality reduction space. For each component axis, split the
    burst activity in a number of features according to the scores of the bursts
    along the axis.

    Then, estimate the burst rate of a given number of channels for each of these
    groups.

    Parameters
    ----------
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    tf_method: str {"wavelets", "superlets}, optional
               String indicating the algorithm used for burst
               extraction.
               Defaults to "superlets".
    significance_threshold: float, optional
                            Fraction of pixels that indicates the lower bound
                            for marking a feature as significant.
                            Defaults to 0.25.
    task_time_lims: two-element list or numpy array, optional
                    Start and end time of the task period (in seconds,
                    relative to the time-locked event). If fet to "None",
                    it will be overwritten by the variables in the corresponding
                    'variables.json' file.
                    Defaults to "None".

    Attributes
    ----------
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
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    baseline_time_lims: two-element list
                        Baseline period beginning and end (in seconds
                        relative to time-locked event).
    task_time_lims: two-element list
                    Task period beginning and end (in seconds
                    relative to time-locked event).
    bin_dt: float
            Step for time axis discretization.
    binned_exp_time: 1D numpy array
                     Discretized, trimmed time axis.
    time_bins: int
               Number of total bins in 'binned_exp_time'.
    """

    def __init__(
        self,
        exp_variables,
        tf_method="superlets",
        significance_threshold=0.25,
        task_time_lims=None,
    ):
        self.exp_variables = exp_variables
        self.tf_method = tf_method
        self.significance_threshold = significance_threshold
        self.task_time_lims = task_time_lims

        # Direct access to variables, and time axis creation.
        self.savepath = exp_variables["dataset_path"]
        self.channels = exp_variables["channels"]

        self.exp_time_periods = exp_variables["exp_time_periods"]
        self.tmin = exp_variables["tmin"]
        self.tmax = exp_variables["tmax"]
        self.sfreq = exp_variables["sfreq"]
        self.exp_time = np.linspace(
            self.tmin,
            self.tmax,
            int((np.abs(self.tmax - self.tmin)) * self.sfreq) + 1,
        )
        self.exp_time = np.around(self.exp_time, decimals=3)

        # Binned time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
             baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        self.baseline_time_lims = [self.exp_time_periods[0], self.exp_time_periods[1]]
        if self.task_time_lims == None:
            self.task_time_lims = [self.exp_time_periods[1], self.exp_time_periods[2]]

        self.bin_dt = 0.10
        erds_time = self.exp_time[baseline_begin : rebound_end + 1]
        binning = np.arange(erds_time[0], erds_time[-1] + self.bin_dt, self.bin_dt)
        self.binned_exp_time = np.around(binning, decimals=2)

        self.time_bins = len(self.binned_exp_time) - 1

    def _slide_win_metric(
        self, sub_dict, channel_ids, metrics, measure_lims, binned_time, bin_dt
    ):
        """
        Compute the burst rate modulation for each channel and trial.

        Parameters
        ----------
        sub_dict: dict
                  Dictionary containing all detected bursts of a subject.
        channel_ids: list
                     Indices of channels to take into account during feature
                     creation.
        metrics: numpy array
                 Array of all burst scores along a given component axis.
        measure_lims: numpy array
                      Array containing the scores corresponding to the limits
                      for splitting a given component axis to a given number
                      of groups.
        binned_time: numpy array
                     Array representing the trimmed experimental time
                     with wide time steps, needed for an estimation
                     of burst rate in non-overlapping windows.
        bin_dt: float
                Time step for advancing to a new time window.

        Returns
        -------
        mv_burst_features: numpy array
                           Features along a given principal component axis.
                           Dimensions: [#trials, #groups, #time bins].
        """

        # Total available trials.
        trials = np.unique(sub_dict["trial"])

        # Initialization of variable that stores features from all channels
        # along a given component axis.
        mv_burst_features = []

        # Find bursts of specific channels and scores.
        for k in channel_ids:
            for m, m_lims in enumerate(measure_lims):
                # Bursts corresponding to each channel.
                area_bursts = np.sum([sub_dict["channel"] == k], axis=0)

                # Keep the bursts with specific scores.
                if m != len(measure_lims) - 1:
                    mc_bursts = np.where(
                        (metrics >= m_lims[0]) & (metrics < m_lims[1]) & (area_bursts)
                    )[0]
                else:
                    mc_bursts = np.where((metrics >= m_lims[0]) & (area_bursts))[0]
                mc_burst_trials = sub_dict["trial"][mc_bursts]
                mc_burst_times = sub_dict["peak_time"][mc_bursts]

                # Group burst rate.
                mv_burst_feature = np.zeros((len(trials), len(binned_time) - 1))
                for t_idx, trial in enumerate(trials):
                    # Trial bursts.
                    t_bursts = np.where(mc_burst_trials == trial)[0]
                    mv_hist, _ = np.histogram(
                        mc_burst_times[t_bursts], bins=binned_time
                    )

                    # Convert count to rate.
                    mv_burst_feature[t_idx, :] = mv_hist / bin_dt

                mv_burst_features.append(mv_burst_feature)

        # Tensor [trials, time, groups] -> [trials, groups, time]
        mv_burst_features = np.array(mv_burst_features)
        mv_burst_features = np.rollaxis(mv_burst_features, 0, 2)

        return mv_burst_features

    def transform(
        self,
        subject_dictionary,
        sub_scores_dists,
        drm_scores_dists,
        labels,
        channel_ids,
        comps_to_analyze,
        comps_groups,
        time_domain=False,
        keep_significant_features=False,
    ):
        """
        Burst rate features creation.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects-specific burst dictionary.
        subject_dictionary: dict
                            Dictionary containing all detected bursts of 'subject'.
        sub_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on all trials) for 'subject'.
        drm_scores_dists: list
                          List containing all burst scores along each component
                          axis (based on trials used while creating the dimensionality
                          reduction model) for 'subject'.
        labels: numpy array
                Array of strings containing the labels for each trial.
        channel_ids: list
                     Indices of channels to take into account during feature
                     creation.
        comps_to_analyze: list or numpy array
                          List of the indices of components used for feature
                          creation.
        comps_groups: int
                      Number of groups the scores of each component axis
                      should be split into.
        time_domain: bool, optional
                     If "True" return burst features as time series of burst
                     modulation. If "False" return a scalar representing the
                     average burst modulation within 'self.task_time_lims'
                     attribute, per trial and channel.
                     Defaults to "False".
        keep_significant_features: bool, optional
                                   If "True", for each principal component axis
                                   only return the features that are marked as
                                   statistically significant, based on a
                                   cluster-based permutation test.
                                   Defaults to "False".

        Returns
        -------
        features: numpy array
                  Burst features. Dimensions are 3D [#trials, #groups,
                  #time bins] or 2D [#trials, #groups] depending on
                  the value of the 'time_domain' parameter.
        labels: numpy array
                Array of strings containing the correct number of labels
                (exclusion of any trials without bursts).
        """

        # Split the burst dictionary in trials corresponding to each condition.
        unique_labels = np.unique(labels)
        exp_trials_cond1 = np.where(labels == unique_labels[0])[0]
        exp_trials_cond2 = np.where(labels == unique_labels[1])[0]
        sub_dict_cond1, sub_dict_trials_cond1 = dict_bycondition(
            subject_dictionary, exp_trials_cond1
        )
        sub_dict_cond2, sub_dict_trials_cond2 = dict_bycondition(
            subject_dictionary, exp_trials_cond2
        )

        bursty_trials_cond1 = np.unique(sub_dict_cond1["trial"])
        bursty_trials_cond2 = np.unique(sub_dict_cond2["trial"])
        all_bursty_trials = np.argsort(
            np.hstack([bursty_trials_cond1, bursty_trials_cond2])
        )

        # Optionally keep the significant features (based on cluster-based permutations tests).
        if keep_significant_features == True:
            gtk = significant_components(
                sub_dict_cond1,
                sub_dict_cond2,
                sub_dict_trials_cond1,
                sub_dict_trials_cond2,
                self.channels,
                sub_scores_dists,
                comps_to_analyze,
                comps_groups,
                self.binned_exp_time,
                self.bin_dt,
                self.task_time_lims,
                self.significance_threshold,
            )

        # Bursts' sliding window features initialization [trials, groups, time].
        # features: [#channels x #pcs_to_analyze x #comps_groups]
        if isinstance(channel_ids, int):
            channel_ids = [channel_ids]
        n_features = len(channel_ids) * len(comps_to_analyze) * comps_groups
        features_cond1 = np.zeros(
            [len(bursty_trials_cond1), n_features, self.time_bins]
        )
        features_cond2 = np.zeros(
            [len(bursty_trials_cond2), n_features, self.time_bins]
        )

        for comp_id, (dsd, scd) in enumerate(zip(drm_scores_dists, sub_scores_dists)):
            # Separate metrics per condition, and use common metric limits per component.
            sub_score_dist_cond1 = scd[np.hstack(sub_dict_trials_cond1)]
            sub_score_dist_cond2 = scd[np.hstack(sub_dict_trials_cond2)]

            if comps_groups == 1:
                raise ValueError(
                    "You need to specify at least 2 groups when creating burst features!"
                )
            else:
                lims = np.linspace(np.min(dsd), np.max(dsd), comps_groups + 1)
                scores_lims = [[lims[i], lims[i + 1]] for i in range(comps_groups)]
                scores_lims = np.array(scores_lims)

            mv_burst_rates_cond1 = self._slide_win_metric(
                sub_dict_cond1,
                channel_ids,
                sub_score_dist_cond1,
                scores_lims,
                self.binned_exp_time,
                self.bin_dt,
            )
            mv_burst_rates_cond2 = self._slide_win_metric(
                sub_dict_cond2,
                channel_ids,
                sub_score_dist_cond2,
                scores_lims,
                self.binned_exp_time,
                self.bin_dt,
            )

            # Fill-in the features tensors.
            n_pc_feats = len(channel_ids) * comps_groups
            start = n_pc_feats * comp_id
            end = start + n_pc_feats

            features_cond1[:, start:end, :] = mv_burst_rates_cond1
            features_cond2[:, start:end, :] = mv_burst_rates_cond2

        # Selection of "informative" features.
        if keep_significant_features == True:
            gtk += (np.array(gtk) + (len(comps_to_analyze) * comps_groups)).tolist()
            features_cond1 = features_cond1[:, gtk, :]
            features_cond2 = features_cond2[:, gtk, :]

        # Keep features in the time window of interest.
        toi = np.where(
            np.logical_and(
                self.binned_exp_time >= self.task_time_lims[0],
                self.binned_exp_time <= self.task_time_lims[1],
            )
        )[0]

        if time_domain == False:
            features_cond1 = np.mean(features_cond1[:, :, toi], axis=-1)
            features_cond2 = np.mean(features_cond2[:, :, toi], axis=-1)
        elif time_domain == True:
            features_cond1 = features_cond1[:, :, toi]
            features_cond2 = features_cond2[:, :, toi]

        # Return correct number of labels per condition.
        labels_cond1 = np.repeat(unique_labels[0], features_cond1.shape[0])
        labels_cond2 = np.repeat(unique_labels[1], features_cond2.shape[0])

        features = np.vstack((features_cond1, features_cond2))[all_bursty_trials]
        labels = np.hstack((labels_cond1, labels_cond2))[all_bursty_trials]

        return features, labels

    def plot_features(
        self,
        subject,
        subject_dictionary,
        sub_scores_dists,
        labels,
        comps_to_analyze,
        comps_to_vis,
        comps_groups,
        solver,
        epochs,
        epochs_power_dict,
        band="beta",
        baseline_correction="independent",
        show_splits=False,
        show_stats=False,
        show_sample=False,
        apply_baseline="True",
        rate_computation="joint",
        plot_difference=False,
        savefigs=True,
        plot_format="pdf",
    ):
        """
        Plot trial-averaged features.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects' data used for creating
                 the burst dictionary.
        subject_dictionary: dict
                            Dictionary containing all detected bursts of 'subject'.
        sub_scores_dists: list
                          List containing all burst scores along each principal
                          component axis (based on all trials) for 'subject'.
        labels: numpy array
                Array of strings containing the labels for each trial.
        comps_to_analyze: list or numpy array
                          List of the indices of components used for feature
                          creation.
        comps_to_vis: list or numpy array
                      List of the indices of components used for feature
                      creation.
        comps_groups: int
                      Number of groups for the scores of each component
                      axis should be split into.
        solver: str, {"pca", "csp"}
                Dimensionality reduction algorithm.
        epochs: MNE-python epochs object
                Epochs object corresponding to 'dataset'.
        epochs_power_dict: dict
                           Dictionary of arrays containing the trial-averaged,
                           baseline-corrected Hilbert envelope power for a given
                           condition, and corresponding sem.
        band: str {"mu", "beta"}, optional
              Select band for burst detection.
              Defaults to "beta".
        baseline_correction: str {"independent", "channel", "condition"}, optional
                             USED FOR FIGURE OF COMPONENT- AND WAVEFORM-RESOLVED BURST RATE.
                             String indicating whether to apply baseline correction
                             per channel and condition, across conditions for each
                             channel or across channels for each condition.
                             Defaults to "independent".
        show_splits: bool, optional
                     If set to "True" show lines that indicate how the scores axis is
                     split in order to create features.
                     Defaults to "False".
        show_stats: bool, optional
                    If set to "True" show contour lines that indicate statistically
                    significant differences per channel/condition according to
                    cluster-based permutation tests.
                    Defaults to "False".
        show_sample: bool, optional
                     If set to "True" produce a child figure with only a single plot
                     for pedagogic purposes.
                     Defaults to "False".
        apply_baseline: bool, optional
                        USED FOR FIGURE OF COMPONENTS AND WAVEFORMS
                        CO-MODULATION OF BURST RATE.
                        If set to "True" apply baseline correction using
                        the average across trials as baseline.
                        Defaults to "True".
        rate_computation: str {"independent", "joint"}, optional
                          String that controls whether the burst rate is
                          computed indepently along each of the provided
                          'comps_to_vis3d', or jointly.
                          Defaults to "joint".
        plot_difference: bool or "condition", optional
                         If "True" plot the difference between C3 and C4 per
                         condition, instead of each channel. If set to
                         "condition" plot the difference of the same channel
                         between conditions.
                         Defaults to "False".
        savefigs: bool, optional
                  If set to "True" the visualizations are automatically
                  saved. If set to "False" they are shown on screen.
                  Defaults to "True".
        plot_format: str {"pdf", "png"}, optional
                     File format. Prefer "pdf" for editing with vector graphics
                     applications, or "png" for less space usage and better
                     integration with presentations.
                     Defaults to "pdf".
        """

        # Split the burst dictionary in trials corresponding to each condition.
        unique_labels = np.unique(labels)
        exp_trials_cond1 = np.where(labels == unique_labels[0])[0]
        exp_trials_cond2 = np.where(labels == unique_labels[1])[0]

        sub_dict_cond1, sub_dict_trials_cond1 = dict_bycondition(
            subject_dictionary, exp_trials_cond1
        )
        sub_dict_cond2, sub_dict_trials_cond2 = dict_bycondition(
            subject_dictionary, exp_trials_cond2
        )

        # Plot the across-trials average modulation of burst rate across
        # each principal component axis.
        if len(comps_to_analyze) <= 5:
            features_modulation(
                subject,
                self.exp_variables,
                sub_dict_cond1,
                sub_dict_cond2,
                sub_dict_trials_cond1,
                sub_dict_trials_cond2,
                self.channels,
                sub_scores_dists,
                comps_to_analyze,
                comps_to_vis,
                comps_groups,
                self.binned_exp_time,
                self.bin_dt,
                self.task_time_lims,
                self.baseline_time_lims,
                self.tf_method,
                band,
                solver,
                epochs,
                epochs_power_dict,
                self.savepath,
                baseline_correction=baseline_correction,
                show_splits=show_splits,
                show_stats=show_stats,
                show_sample=show_sample,
                savefigs=savefigs,
                plot_format=plot_format,
            )

        else:
            if len(comps_to_analyze) % 5 == 0:
                total_chunks = len(comps_to_analyze) // 5
                split = 5
            else:
                total_chunks = len(comps_to_analyze) // 5 + 1
                split = int(np.ceil(len(comps_to_analyze) / total_chunks))

            for chunk in range(total_chunks):
                some_comps_to_analyze = comps_to_analyze[
                    split * chunk : split * (chunk + 1)
                ]
                some_sub_scores_dists = sub_scores_dists[
                    split * chunk : split * (chunk + 1)
                ]
                features_modulation(
                    subject,
                    dataset,
                    self.exp_variables,
                    sub_dict_cond1,
                    sub_dict_cond2,
                    sub_dict_trials_cond1,
                    sub_dict_trials_cond2,
                    self.channels,
                    some_sub_scores_dists,
                    some_comps_to_analyze,
                    comps_to_vis,
                    comps_groups,
                    self.binned_exp_time,
                    self.bin_dt,
                    self.task_time_lims,
                    self.baseline_time_lims,
                    self.tf_method,
                    band,
                    solver,
                    epochs,
                    epochs_power_dict,
                    self.savepath,
                    baseline_correction=baseline_correction,
                    show_splits=show_splits,
                    show_stats=show_stats,
                    show_sample=show_sample,
                    savefigs=savefigs,
                    plot_format=plot_format,
                )

        # Plot co-modulation of burst rate across different components.
        features_modulation_nd(
            subject,
            self.exp_variables,
            sub_dict_cond1,
            sub_dict_cond2,
            sub_dict_trials_cond1,
            sub_dict_trials_cond2,
            self.channels,
            sub_scores_dists,
            comps_to_vis,
            comps_groups,
            self.binned_exp_time,
            self.bin_dt,
            band,
            self.task_time_lims,
            self.baseline_time_lims,
            self.savepath,
            apply_baseline=apply_baseline,
            rate_computation=rate_computation,
            plot_difference=plot_difference,
            savefigs=savefigs,
            plot_format=plot_format,
        )


class TfFeatures:
    """
    Create features of burst rate modulation in time, suitable for binary
    classification tasks.

    Given the burst activity of a single subject, split the burst activity in a
    number of features according to the distribution of the bursts along each
    characteristic of interest.

    Then, estimate the burst rate of a given number of channels for each of these
    groups.

    Parameters
    ----------
    exp_variabeles: dict
                    Experimental variables contained in the corresponding
                    'variables.json' file.
    tf_method: str {"wavelets", "superlets}, optional
               String indicating the algorithm used for burst
               extraction.
               Defaults to "superlets".
    significance_threshold: float, optional
                            Fraction of pixels that indicates the lower bound
                            for marking a feature as significant.
                            Defaults to 0.25.
    task_time_lims: two-element list or numpy array, optional
                    Start and end time of the task period (in seconds,
                    relative to the time-locked event). If fet to "None",
                    it will be overwritten by the variables in the corresponding
                    'variables.json' file.
                    Defaults to "None".

    Attributes
    ----------
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
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    baseline_time_lims: two-element list
                        Baseline period beginning and end (in seconds
                        relative to time-locked event).
    task_time_lims: two-element list
                    Task period beginning and end (in seconds
                    relative to time-locked event).
    bin_dt: float
            Step for time axis discretization.
    binned_exp_time: 1D numpy array
                     Discretized, trimmed time axis.
    time_bins: int
               Number of total bins in 'binned_exp_time'.
    """

    def __init__(
        self,
        exp_variables,
        tf_method="superlets",
        significance_threshold=0.25,
        task_time_lims=None,
    ):
        self.exp_variables = exp_variables
        self.tf_method = tf_method
        self.significance_threshold = significance_threshold
        self.task_time_lims = task_time_lims

        # Direct access to variables, and time axis creation.
        self.savepath = exp_variables["dataset_path"]
        self.channels = exp_variables["channels"]

        self.exp_time_periods = exp_variables["exp_time_periods"]
        self.tmin = exp_variables["tmin"]
        self.tmax = exp_variables["tmax"]
        self.sfreq = exp_variables["sfreq"]
        self.exp_time = np.linspace(
            self.tmin,
            self.tmax,
            int((np.abs(self.tmin) + np.abs(self.tmax)) * self.sfreq) + 1,
        )
        self.exp_time = np.around(self.exp_time, decimals=3)

        # Binned time axis.
        try:
            baseline_begin = int(np.where(self.exp_time == self.exp_time_periods[0])[0])
        except:
             baseline_begin = np.where(self.exp_time >= self.exp_time_periods[0])[0][0]
        try:
            rebound_end = int(np.where(self.exp_time == self.exp_time_periods[3])[0])
        except:
            rebound_end = np.where(self.exp_time <= self.exp_time_periods[3])[0][-1]

        self.baseline_time_lims = [self.exp_time_periods[0], self.exp_time_periods[1]]
        if self.task_time_lims == None:
            self.task_time_lims = [self.exp_time_periods[1], self.exp_time_periods[2]]

        self.bin_dt = 0.10
        erds_time = self.exp_time[baseline_begin : rebound_end + 1]
        binning = np.arange(erds_time[0], erds_time[-1] + self.bin_dt, self.bin_dt)
        self.binned_exp_time = np.around(binning, decimals=2)

        self.time_bins = len(self.binned_exp_time) - 1

    def _slide_win_metric(
        self, sub_dict, channel_ids, metrics, measure_lims, binned_time, bin_dt
    ):
        """
        Compute the burst rate modulation for each channel and trial.

        Parameters
        ----------
        sub_dict: dict
                  Dictionary containing all detected bursts of a subject.
        channel_ids: list
                     Indices of channels to take into account during feature
                     creation.
        metrics: numpy array
                 Array of all burst scores along a given component axis.
        measure_lims: numpy array
                      Array containing the scores corresponding to the limits
                      for splitting a given component axis to a given number
                      of groups.
        binned_time: numpy array
                     Array representing the trimmed experimental time
                     with wide time steps, needed for an estimation
                     of burst rate in non-overlapping windows.
        bin_dt: float
                Time step for advancing to a new time window.

        Returns
        -------
        mv_burst_features: numpy array
                           Features along a given principal component axis.
                           Dimensions: [#trials, #groups, #time bins].
        """

        # Total available trials.
        trials = np.unique(sub_dict["trial"])

        # Initialization of variable that stores features from all channels
        # along a given component axis.
        mv_burst_features = []

        # Find bursts of specific channels and scores.
        for k in channel_ids:
            for m, m_lims in enumerate(measure_lims):
                # Bursts corresponding to each channel.
                area_bursts = np.sum([sub_dict["channel"] == k], axis=0)

                # Keep the bursts with specific scores.
                if m != len(measure_lims) - 1:
                    mc_bursts = np.where(
                        (metrics >= m_lims[0]) & (metrics < m_lims[1]) & (area_bursts)
                    )[0]
                else:
                    mc_bursts = np.where((metrics >= m_lims[0]) & (area_bursts))[0]

                mc_burst_trials = sub_dict["trial"][mc_bursts]
                mc_burst_times = sub_dict["peak_time"][mc_bursts]

                # Group burst rate.
                mv_burst_feature = np.zeros((len(trials), len(binned_time) - 1))
                for t_idx, trial in enumerate(trials):
                    # Trial bursts.
                    t_bursts = np.where(mc_burst_trials == trial)[0]
                    mv_hist, _ = np.histogram(
                        mc_burst_times[t_bursts], bins=binned_time
                    )

                    # Convert count to rate.
                    mv_burst_feature[t_idx, :] = mv_hist / bin_dt

                mv_burst_features.append(mv_burst_feature)

        # Tensor [trials, time, groups] -> [trials, groups, time]
        mv_burst_features = np.array(mv_burst_features)
        mv_burst_features = np.rollaxis(mv_burst_features, 0, 2)

        return mv_burst_features

    def transform(
        self,
        subject_dictionary,
        sub_chars_dists,
        labels,
        channel_ids,
        chars_to_analyze,
        chars_groups,
        time_domain=False,
    ):
        """
        Burst rate features creation.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects-specific burst dictionary.
        subject_dictionary: dict
                            Dictionary containing all detected bursts of 'subject'.
        sub_chars_dists: list
                         List containing all burst scores for each burst dictionary
                         charecteristic (based on all trials) for 'subject'.
        labels: numpy array
                Array of strings containing the labels for each trial.
        channel_ids: list
                     Indices of channels to take into account during feature
                     creation.
        chars_to_analyze: list or numpy array
                          List of the indices of characteristics used for feature
                          creation.
        chars_groups: int
                  Number of groups for the scores of burst characteristic axis
                  should be split into.
        time_domain: bool, optional
                     If "True" return burst features as time series of burst
                     modulation. If "False" return a scalar representing the
                     average burst modulation within 'self.task_time_lims'
                     attribute, per trial and channel.
                     Defaults to "False".
        keep_significant_features: bool, optional
                                   If "True", for each principal component axis
                                   only return the features that are marked as
                                   statistically significant, based on a
                                   cluster-based permutation test.
                                   Defaults to "False".

        Returns
        -------
        features: numpy array
                  Burst features. Dimensions are 3D [#trials, #groups,
                  #time bins] or 2D [#trials, #groups] depending on
                  the value of the 'time_domain' parameter.
        labels: numpy array
                Array of strings containing the correct number of labels
                (exclusion of any trials without bursts).
        """

        # Split the burst dictionary in trials corresponding to each condition.
        unique_labels = np.unique(labels)
        exp_trials_cond1 = np.where(labels == unique_labels[0])[0]
        exp_trials_cond2 = np.where(labels == unique_labels[1])[0]
        sub_dict_cond1, sub_dict_trials_cond1 = dict_bycondition(
            subject_dictionary, exp_trials_cond1
        )
        sub_dict_cond2, sub_dict_trials_cond2 = dict_bycondition(
            subject_dictionary, exp_trials_cond2
        )

        bursty_trials_cond1 = np.unique(sub_dict_cond1["trial"])
        bursty_trials_cond2 = np.unique(sub_dict_cond2["trial"])
        all_bursty_trials = np.argsort(
            np.hstack([bursty_trials_cond1, bursty_trials_cond2])
        )

        # Bursts' sliding window features initialization [trials, groups, time].
        # features: [#channels x #chars_to_analyze x #chars_groups]
        if isinstance(channel_ids, int):
            channel_ids = [channel_ids]
        n_features = len(channel_ids) * len(chars_to_analyze) * chars_groups
        features_cond1 = np.zeros(
            [len(bursty_trials_cond1), n_features, self.time_bins]
        )
        features_cond2 = np.zeros(
            [len(bursty_trials_cond2), n_features, self.time_bins]
        )

        for c, char_dist in enumerate(
            sub_chars_dists[chars_to_analyze[0] - 1 : chars_to_analyze[-1]]
        ):
            # Separate metrics per condition, and use common metric limits per characteristic.
            sub_score_dist_cond1 = char_dist[np.hstack(sub_dict_trials_cond1)]
            sub_score_dist_cond2 = char_dist[np.hstack(sub_dict_trials_cond2)]

            # Limits in the metrics that are used to split each image in features.
            if chars_groups == 1:
                raise ValueError(
                    "You need to specify at least 2 groups when creating features!"
                )
            else:
                lims = np.linspace(
                    np.min(char_dist), np.max(char_dist), chars_groups + 1
                )
                scores_lims = [[lims[i], lims[i + 1]] for i in range(chars_groups)]
                scores_lims = np.array(scores_lims)

            mv_burst_rates_cond1 = self._slide_win_metric(
                sub_dict_cond1,
                channel_ids,
                sub_score_dist_cond1,
                scores_lims,
                self.binned_exp_time,
                self.bin_dt,
            )
            mv_burst_rates_cond2 = self._slide_win_metric(
                sub_dict_cond2,
                channel_ids,
                sub_score_dist_cond2,
                scores_lims,
                self.binned_exp_time,
                self.bin_dt,
            )

            # Fill-in the features tensors.
            n_pc_feats = len(channel_ids) * chars_groups
            start = n_pc_feats * c
            end = start + n_pc_feats

            features_cond1[:, start:end, :] = mv_burst_rates_cond1
            features_cond2[:, start:end, :] = mv_burst_rates_cond2

        # Keep features in the time window of interest.
        toi = np.where(
            np.logical_and(
                self.binned_exp_time >= self.task_time_lims[0],
                self.binned_exp_time <= self.task_time_lims[1],
            )
        )[0]

        if time_domain == False:
            features_cond1 = np.mean(features_cond1[:, :, toi], axis=-1)
            features_cond2 = np.mean(features_cond2[:, :, toi], axis=-1)
        elif time_domain == True:
            features_cond1 = features_cond1[:, :, toi]
            features_cond2 = features_cond2[:, :, toi]

        # Return correct number of labels per condition.
        labels_cond1 = np.repeat(unique_labels[0], features_cond1.shape[0])
        labels_cond2 = np.repeat(unique_labels[1], features_cond2.shape[0])

        features = np.vstack((features_cond1, features_cond2))[all_bursty_trials]
        labels = np.hstack((labels_cond1, labels_cond2))[all_bursty_trials]

        return features, labels

    def plot_features(
        self,
        subject,
        dataset,
        subject_dictionary,
        sub_chars_dists,
        labels,
        epochs,
        chars_groups,
        band_pass,
        band="beta",
        show_splits=False,
        show_stats=False,
        savefigs=True,
        plot_format="pdf",
    ):
        """
        Plot trial-averaged features.

        Parameters
        ----------
        subject: int
                 Integer indicating the subjects' data used for creating
                 the burst dictionary.
        dataset: MOABB object
                 Dataset from the MOABB for the analysis.
        subject_dictionary: dict
                            Dictionary containing all detected bursts of 'subject'.
        sub_chars_dists: list
                         List containing all burst scores for each burst dictionary
                         charecteristic (based on all trials) for 'subject'.
        labels: numpy array
                Array of strings containing the labels for each trial.
        epochs: MNE-python epochs object
                Epochs object corresponding to 'dataset'.
        chars_groups: int
                  Number of groups for the scores of burst characteristic axis
                  should be split into.
        band_pass: two-element list or numpy array
                   Band pass limits for filtering the data while loading them.
        band: str {"mu", "beta"}, optional
              Select band for burst detection.
              Defaults to "beta".
        show_splits: bool, optional
                     If set to "True" show lines that indicate how the scores axis is
                     split in order to create features.
                     Defaults to "False".
        show_stats: bool, optional
                    If set to "True" show contour lines that indicate statistically
                    significant differences per channel/condition according to
                    cluster-based permutation tests.
                    Defaults to "False".
        savefigs: bool, optional
                  If set to "True" the visualizations are automatically
                  saved. If set to "False" they are shown on screen.
                  Defaults to "True".
        plot_format: str {"pdf", "png"}, optional
                     File format. Prefer "pdf" for editing with vector graphics
                     applications, or "png" for less space usage and better
                     integration with presentations.
                     Defaults to "pdf".
        """

        # Split the burst dictionary in trials corresponding to each condition.
        unique_labels = np.unique(labels)
        exp_trials_cond1 = np.where(labels == unique_labels[0])[0]
        exp_trials_cond2 = np.where(labels == unique_labels[1])[0]

        sub_dict_cond1, sub_dict_trials_cond1 = dict_bycondition(
            subject_dictionary, exp_trials_cond1
        )
        sub_dict_cond2, sub_dict_trials_cond2 = dict_bycondition(
            subject_dictionary, exp_trials_cond2
        )

        # Plot the across-trials average modulation of burst rate across
        # each burst characteristic.
        chars_modulation(
            subject,
            dataset,
            self.exp_variables,
            sub_dict_cond1,
            sub_dict_cond2,
            sub_dict_trials_cond1,
            sub_dict_trials_cond2,
            self.channels,
            sub_chars_dists,
            chars_groups,
            band_pass,
            self.binned_exp_time,
            self.bin_dt,
            self.task_time_lims,
            self.baseline_time_lims,
            self.tf_method,
            band,
            epochs,
            self.savepath,
            show_splits=show_splits,
            show_stats=show_stats,
            savefigs=savefigs,
            plot_format=plot_format,
        )
