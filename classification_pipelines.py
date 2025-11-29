"""
Collection of simple classification pipelines suitable for
different type of data.

Results are automatically saved to the directory of each
dataset.
"""

import numpy as np
from os.path import join

from mne.filter import filter_data
from scipy.signal import hilbert

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

from burst_space import BurstSpace
from burst_features import BurstFeatures, TfFeatures
from preprocess import load_sub, apply_preprocessing


def classify_waveforms(
    subjects,
    exp_variables,
    channel_ids,
    savepath,
    classifier,
    trials_fraction=0.2,
    solver="csp",
    band="beta",
    remove_fooof=True,
    metric="rocauc",
    reps=100,
    stratification=5,
    n_comps=8,
    n_comps_tk=8,
    n_groups=9,
    n_folds=5,
    keep_significant_features=False,
    limit_hspace=False,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based on burst waveform
    features.

    Optionally, adopt a stratification approach for creating the burst space model,
    in order to eliminate the randomness introduced in the results by the specific
    trials that have been used each time during this step.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    channel_ids: list
                 Indices of channels to take into account during burst
                 dictionary creation.
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    classifier: scikit-learn model
                Any scikit learn model that can be used as a classifier
                (e.g. LDA, SVC).
    trials_fraction: float, optional
                     Fraction of total trials in each subject's data
                     used to create the burst dictionary.
                     Defaults to 0.2.
    solver: str {"pca", "csp"}, optional
            Dimensionality reduction algorithm. Implements the PCA sklearn model,
            or the MNE-python CSP model.
            Defaults to "pca".
    band: str {"mu", "beta"}, optional
              Select band for burst detection.
              Defaults to "beta".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    stratification: int, optional
                    Split the dictionary's trials in (almost) equal strata;
                    the assumption is that you are interested in a repeated
                    estimation of the dictionary and the dimensionality
                    reduction model. If set to 1, assumes that can draw a
                    random sample.
                    Defaults to 5.
    n_comps: int or float, optional
             Number of components to retrieve, or if set to 'float', explained
             variance when 'solver' is set to PCA.
             Defaults to 8.
    n_comps_tk: int, optional
                Maximum number of components to explore.
                Defaults to 8.
    n_groups: int, optional
              Number of groups for the scores of each component
              axis should be split into.
              Defaults to 9.
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    keep_significant_features: bool, optional
                               If "True", for each principal component axis
                               only return the features that are marked as
                               statistically significant, based on a
                               cluster-based permutation test.
                               Defaults to "False".
    limit_hspace: bool, optional
                  If set to "True" restricts the grid search of the parameter
                  space to a single point.
                  Defaults to "False".
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # Initialization of variables for storing results.
    if limit_hspace == False:
        mean_accs = np.empty([len(subjects), stratification, n_groups - 1, n_comps_tk])
        std_accs = np.empty([len(subjects), stratification, n_groups - 1, n_comps_tk])
    elif limit_hspace == True:
        mean_accs = np.empty([len(subjects), stratification])
        std_accs = np.empty([len(subjects)])

    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    for subject_id, subject in enumerate(subjects):
        print("Estimating decoding accuracy for subject {}...".format(subject))

        # Subject-specific training labels.
        labels = np.load(join(savepath, "sub_{}/labels.npy".format(subject)))

        # Stratification.
        for i in range(stratification):
            # Burst space model.
            if stratification == 1:
                bspace = BurstSpace(
                    exp_variables,
                    subjects,
                    trials_fraction=trials_fraction,
                    channel_ids=channel_ids,
                    remove_fooof=remove_fooof,
                    band=band,
                )
            else:
                bspace = BurstSpace(
                    exp_variables,
                    subjects,
                    trials_fraction=trials_fraction,
                    channel_ids=channel_ids,
                    remove_fooof=remove_fooof,
                    band=band,
                    stratification=stratification,
                    strata_id=i,
                )
            bspace.fit_transform(solver=solver, n_components=n_comps)
            drm_trials = bspace.drm_trials

            # Hyper-parameters exploration.

            # Define the parameter space to be explored.
            # At least two features per component are required.
            if limit_hspace == False:
                min_groups = 2
                max_groups = n_groups + 1
            elif limit_hspace == True:
                min_groups = 2 + n_groups[subject_id][i]
                max_groups = 2 + n_groups[subject_id][i] + 1

            for comp_groups in range(min_groups, max_groups):
                gid = comp_groups - 2

                # Begin exploration using only the first component.
                if limit_hspace == False:
                    cta = [1]
                    stop_crit = n_comps_tk
                elif limit_hspace == True:
                    cta = np.arange(1, n_comps_tk[subject_id][i] + 2, 1).tolist()
                    stop_crit = n_comps_tk[subject_id][i]

                while True:
                    # Compute burst features.
                    (
                        subject_dictionary,
                        sub_scores_dists,
                        drm_scores_dists,
                    ) = bspace.transform_sub(subject, cta)
                    bclf = BurstFeatures(exp_variables)
                    b_features, b_labels = bclf.transform(
                        subject_dictionary,
                        sub_scores_dists,
                        drm_scores_dists,
                        labels,
                        channel_ids,
                        cta,
                        comp_groups,
                        keep_significant_features=keep_significant_features,
                    )

                    if limit_hspace == False:
                        # Remove trials used in the dictionary.
                        b_features = np.delete(
                            b_features, drm_trials[subject_id], axis=0
                        )
                        b_labels = np.delete(b_labels, drm_trials[subject_id])

                        # Shuffle the data.
                        all_sfl = np.arange(0, b_features.shape[0], 1)

                        tt_features = np.copy(b_features)[all_sfl, :]
                        tt_labels = np.copy(b_labels)[all_sfl]

                        # Initialize an array to store the results per repetition.
                        accuracies = np.zeros([reps, n_folds])

                        # Repeated cross-validation.
                        for r in range(reps):
                            np.random.shuffle(all_sfl)
                            tt_features = tt_features[all_sfl, :]
                            tt_labels = tt_labels[all_sfl]

                            if metric == "accuracy":
                                accuracies[r, :] = (
                                    cross_val_score(
                                        classifier,
                                        tt_features,
                                        tt_labels,
                                        cv=n_folds,
                                        n_jobs=n_jobs,
                                    )
                                    * 100
                                )
                            elif metric == "rocauc":
                                accuracies[r, :] = cross_val_score(
                                    classifier,
                                    tt_features,
                                    tt_labels,
                                    scoring="roc_auc",
                                    cv=n_folds,
                                    n_jobs=n_jobs,
                                )

                        # Store intermediate results.
                        ma = np.mean(accuracies)
                        sa = np.std(accuracies)
                        if limit_hspace == False:
                            mean_accs[subject_id, i, gid, cta[-1] - 1] = ma
                            std_accs[subject_id, i, gid, cta[-1] - 1] = sa
                        elif limit_hspace == True:
                            mean_accs[subject_id, i] = ma
                            std_accs[subject_id, i] = sa

                        # Repeat up to the number of components we are interested in.
                        if cta[-1] < stop_crit:
                            cta.append(cta[-1] + 1)
                        else:
                            break

                    elif limit_hspace == True:
                        # Trials of validation set.
                        val_ids = bspace.val_ids[subject_id]
                        val_trials = drm_trials[subject_id][val_ids]

                        val_features = np.copy(b_features)[val_trials, :]
                        val_labels = np.copy(b_labels)[val_trials]

                        # Remove trials used in the dictionary.
                        bb_features = np.delete(
                            np.copy(b_features), drm_trials[subject_id], axis=0
                        )
                        bb_labels = np.delete(np.copy(b_labels), drm_trials[subject_id])

                        # Train classifier with optimal hyperparameters and test
                        # on validation set.
                        classifier.fit(bb_features, bb_labels)

                        # Store results.
                        mean_accs[subject_id, i] = roc_auc_score(
                            val_labels, classifier.predict_proba(val_features)[:, 1]
                        )
                        break
        if limit_hspace == True:
            std_accs[subject_id] = np.std(mean_accs[subject_id, :])

        print("\n")

    # Save results.
    if limit_hspace == False:
        np.save(
            join(
                savepath,
                "mean_{}_stratified_{}_bursts_{}{}".format(
                    metric, band, solver, fooof_save_str
                ),
            ),
            mean_accs,
        )
        np.save(
            join(
                savepath,
                "std_{}_stratified_{}_bursts_{}{}".format(
                    metric, band, solver, fooof_save_str
                ),
            ),
            std_accs,
        )
    elif limit_hspace == True:
        np.save(
            join(
                savepath,
                "mean_{}_stratified_{}_bursts_{}{}_sel".format(
                    metric, band, solver, fooof_save_str
                ),
            ),
            mean_accs,
        )
        np.save(
            join(
                savepath,
                "std_{}_stratified_{}_bursts_{}{}_sel".format(
                    metric, band, solver, fooof_save_str
                ),
            ),
            std_accs,
        )


def classify_feature(
    subjects,
    exp_variables,
    channel_ids,
    savepath,
    pipe,
    pipeline,
    compute_feature="rate",
    band="beta",
    remove_fooof=True,
    threshold_feature=None,
    percentile=75,
    metric="rocauc",
    reps=100,
    classification_mode="trial",
    n_folds=5,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based overall
    burst rate.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    channel_ids: list
                 Indices of channels to keep for the analysis.
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    pipe: str {"simple", "csp"}
          Naming convention that reflects 'pipeline' for storing results.
    pipeline: scikit-learn model
              Any scikit learn pipeline of models.
              (e.g. CSP & LDA, SVC).
    compute_feature: str {"rate", "amplitude", "volume", "duration", "fr_span",
                     "peak_fr", "cycles"}, optional
                     Feature to examine. "Rate" computes the burst count or rate,
                     depending on the 'rate_normalization' parameter; "amplitude"
                     and "volume" compute the corresponding sum of values; "duration",
                     "fr_span", "peak_fr" and "cycles" compute the corresponding average
                     value.
                     Defaults to "rate".
    band: str {"mu", "beta"}, optional
          Select band for burst detection.
          Defaults to "beta".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    threshold_feature: str or None {"amplitude", "volume", "duration", "cycles",
                                   "fr_span", "peak_time", "peak_fr", None}, optional
                       If no set to "None" only use bursts with a feature
                       above a certain percentile. See the '_feature_threshold'.
                       Defaults to None.
    percentile: int, optional
                Percentile of feature distribution for bursts in the dictionary.
                Any burst below this percentile will not be considered.
                Defaults to 75.
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    classification mode: str {"trial", "incremental"}, optional
                         Choice of time-resolved or full trial decoding scheme.
                         Defaults to "trial".
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # ----- #
    # Initialization of variables for storing results.
    if classification_mode == "trial":
        time_res_str = "_trial"

        mean_accs = np.zeros([len(subjects), reps, n_folds])
        std_accs = np.zeros([len(subjects)])

    elif classification_mode == "incremental":
        time_res_str = "_tr"

        # Binned time axis.
        tmin = exp_variables["tmin"]
        tmax = exp_variables["tmax"]
        exp_time_periods = exp_variables["exp_time_periods"]
        sfreq = exp_variables["sfreq"]
        bin_dt = exp_variables["bin_dt"]
        exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)

        exp_time = np.around(exp_time, decimals=3)

        task_time_lims = [exp_time_periods[1], exp_time_periods[2]]

        task_begin = int(np.where(exp_time == task_time_lims[0] - 0.2)[0])
        task_end = int(np.where(exp_time == task_time_lims[1] + 0.5)[0])
        erds_time = exp_time[task_begin : task_end + 1]

        baseline_begin = int(np.where(exp_time == exp_time_periods[0])[0])
        trial_end = int(np.where(exp_time == exp_time_periods[-1])[0])
        exp_time = exp_time[baseline_begin : trial_end + 1]

        binning = np.arange(erds_time[0], erds_time[-1] + bin_dt, bin_dt)
        binned_erds_time = np.around(binning, decimals=2)

        binning_et = np.arange(exp_time[0], exp_time[-1], bin_dt)
        binned_exp_time = np.around(binning_et, decimals=2)

        # Time windows.
        baseline_window_end = np.where(binned_exp_time <= binned_erds_time[0])[0][-1]
        rebound_window_end = np.where(binned_exp_time <= binned_erds_time[-1])[0][-1]
        n_windows = len(binned_exp_time[baseline_window_end:rebound_window_end]) + 1

        mean_accs = np.zeros([len(subjects), n_windows, reps, n_folds])
        std_accs = np.zeros([len(subjects), n_windows])

    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    for s, subject in enumerate(subjects):
        print(
            "Estimating decoding accuracy for subject {}, feature: {}...".format(
                subject, compute_feature
            ),
        )

        # Suject-specific directory.
        sub_dir = join(savepath, "sub_{}/".format(subject))

        # Labels.
        labels_orig = np.load(join(sub_dir, "labels.npy"))

        # Burst space model.
        bspace = BurstSpace(
            exp_variables,
            subject,
            trials_fraction=1.0,
            channel_ids=channel_ids,
            remove_fooof=remove_fooof,
            band=band,
            threshold_feature=threshold_feature,
            percentile=percentile,
        )

        # Burst rate for all trials.
        if classification_mode == "trial":
            if pipe == "simple":
                return_average = True
            elif pipe == "csp":
                return_average = False

        elif classification_mode == "incremental":
            return_average = False

        burst_feature = bspace.compute_feature_modulation(
            compute_feature=compute_feature,
            rate_normalization=True,
            return_average=return_average,
        )
        missing_trials = bspace.missing_trials

        # Delete labels from missing trials.
        labels_orig = np.delete(labels_orig, missing_trials[0])

        all_sfl = np.arange(0, burst_feature.shape[0], 1)

        if classification_mode == "trial":
            # Repeated cross-validation.
            for rep in range(reps):
                np.random.shuffle(all_sfl)

                features = np.copy(burst_feature)[all_sfl, :]
                labels = np.copy(labels_orig)[all_sfl]

                if pipe == "simple":
                    if metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline, features, labels, cv=n_folds, n_jobs=n_jobs
                            )
                            * 100
                        )
                    elif metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline,
                            features,
                            labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )
                elif pipe == "csp":
                    if metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline, features, labels, cv=n_folds, n_jobs=n_jobs
                            )
                            * 100
                        )
                    elif metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline,
                            features,
                            labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )

        elif classification_mode == "incremental":
            for n in range(n_windows):
                window_end = binned_exp_time[baseline_window_end + n]
                if window_end <= exp_time_periods[1]:
                    # Baseline activity.
                    toi = np.where(binned_exp_time <= window_end)[0]
                else:
                    # Do not take into account baseline activity.
                    toi = np.where(
                        np.logical_and(
                            binned_exp_time >= task_time_lims[0],
                            binned_exp_time <= window_end,
                        )
                    )[0]
                print("\tTime window: {}".format(window_end))

                # Repeated cross-validation.
                for rep in range(reps):
                    np.random.shuffle(all_sfl)

                    features = np.copy(burst_feature)[all_sfl, :]
                    labels = np.copy(labels_orig)[all_sfl]

                    if pipe == "simple":
                        if metric == "accuracy":
                            mean_accs[s, n, rep, :] = (
                                cross_val_score(
                                    pipeline,
                                    np.mean(features[:, :, toi], axis=-1),
                                    labels,
                                    cv=n_folds,
                                    n_jobs=n_jobs,
                                )
                                * 100
                            )
                        elif metric == "rocauc":
                            mean_accs[s, n, rep, :] = cross_val_score(
                                pipeline,
                                np.mean(features[:, :, toi], axis=-1),
                                labels,
                                scoring="roc_auc",
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )
                    elif pipe == "csp":
                        if metric == "accuracy":
                            mean_accs[s, n, rep, :] = (
                                cross_val_score(
                                    pipeline,
                                    features[:, :, toi],
                                    labels,
                                    cv=n_folds,
                                    n_jobs=n_jobs,
                                )
                                * 100
                            )
                        elif metric == "rocauc":
                            mean_accs[s, n, rep, :] = cross_val_score(
                                pipeline,
                                features[:, :, toi],
                                labels,
                                scoring="roc_auc",
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )

                std_accs[s, n] = np.std(mean_accs[s, n, :, :])

        print("\n")

    # Save results.
    if classification_mode == "trial":
        std_accs = np.std(mean_accs, axis=(1, 2))

    np.save(
        join(
            savepath,
            "mean_{}_{}_{}_{}{}".format(
                metric, band, compute_feature, pipe, fooof_save_str
            ),
        ),
        mean_accs,
    )
    np.save(
        join(
            savepath,
            "std_{}_{}_{}_{}{}".format(
                metric, band, compute_feature, pipe, fooof_save_str
            ),
        ),
        std_accs,
    )


def classify_power(
    subjects,
    dataset,
    dataset_name,
    exp_variables,
    band,
    channels,
    channel_ids,
    zapit,
    noise_freq,
    noise_wins,
    savepath,
    pipe,
    pipeline,
    tf_method="superlets",
    remove_fooof=True,
    metric="rocauc",
    reps=100,
    n_folds=5,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based on frequency
    band power.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    dataset: MOABB object
             Dataset from the MOABB prject for the analysis.
    dataset_name: str
                  Corresponding ddtaset name.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    band: str {"beta", "mu", "mu_beta"}
              Frequency band for band-pass filter before estimating power.
    channels: list
              Names of channels to keep while pre-processing the 'dataset'.
    channel_ids: list
                 Indices of channels to keep for the analysis.
    zapit: bool
           If set to "True", iteratively remove a noise artifact from the raw
           signal. The frequency of the artifact is provided by 'this_freq'.
    noise_freq: int or None
               When set to "int", frequency containing power line noise, or
               equivalent artifact. Only considered if 'zapit' is "True".
    noise_wins: list or None
                Window sizes for removing line noise.  Only considered if
                'zapit' is "True".
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    pipe: str {"simple", "csp"}
          Naming convention that reflects 'pipeline' for storing results.
    pipeline: scikit-learn model
              Any scikit learn pipeline of models.
              (e.g. CSP & LDA, SVC).
    tf_method: str {"wavelets", "superlets"}, optional
               String indicating the algorithm used for burst
               extraction.
               Defaults to "superlets".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # ----- #
    # Time period of task.
    tmin = exp_variables["tmin"]
    tmax = exp_variables["tmax"]
    exp_time_periods = exp_variables["exp_time_periods"]
    sfreq = exp_variables["sfreq"]

    exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
    task_time_lims = [exp_time_periods[1], exp_time_periods[2]]
    toi = np.where(
        np.logical_and(exp_time >= task_time_lims[0], exp_time <= task_time_lims[1])
    )[0]

    # Initialization of variables for storing results.
    mean_accs = np.zeros([len(subjects), reps, n_folds])
    std_accs = np.zeros([len(subjects)])

    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    for s, subject in enumerate(subjects):
        print("Estimating decoding accuracy for subject {}...".format(subject))

        # Suject-specific directory.
        sub_dir = join(savepath, "sub_{}/".format(subject))

        # Frequency bands of interest.
        if remove_fooof == False:
            if band == "beta":
                power_band = np.tile([15, 30], (len(channels), 1))
            elif band == "mu":
                power_band = np.tile([8, 15], (len(channels), 1))
            elif band == "mu_beta":
                power_band = np.tile([8, 30], (len(channels, 1)))

        elif remove_fooof == True:
            if band == "beta":
                custom_betas = np.load(
                    join(sub_dir, "beta_bands_{}.npy".format(tf_method))
                )
                for j, custom_beta in enumerate(custom_betas):
                    # Ascertain valid ranges.
                    if np.isnan(custom_beta[0]):
                        custom_betas[j][0] = 15
                    if np.isnan(custom_beta[1]):
                        custom_betas[j][1] = 30
                power_band = custom_betas

            elif band == "mu":
                custom_mus = np.load(join(sub_dir, "mu_bands_{}.npy".format(tf_method)))
                for j, custom_mu in enumerate(custom_mus):
                    # Ascertain valid ranges.
                    if np.isnan(custom_mu[0]):
                        custom_mus[j][0] = 8
                    if np.isnan(custom_mu[1]):
                        custom_mus[j][1] = 13
                power_band = custom_mus

            elif band == "mu_beta":
                custom_betas = np.load(
                    join(sub_dir, "beta_bands_{}.npy".format(tf_method))
                )
                custom_mus = np.load(join(sub_dir, "mu_bands_{}.npy".format(tf_method)))

                power_band = []
                for custom_beta, custom_mu in zip(custom_betas, custom_mus):
                    custom_band = np.array(
                        [np.nanmin(custom_mu), np.nanmax(custom_beta)]
                    )
                    # Ascertain valid ranges.
                    if np.isnan(custom_band[0]):
                        custom_band[0] = 8
                    if np.isnan(custom_band[1]):
                        custom_band[1] = 30
                    power_band.append(custom_band)

        # Subject's raw data loading.
        print("Loading raw data...")

        if dataset_name == "weibo2014":
            epochs, labels, meta = load_sub(
                subject,
                dataset,
                tmin,
                tmax,
                exp_time_periods[:2],
                savepath,
                band_pass=[0, 90],
            )
        else:
            epochs, labels, meta = load_sub(
                subject, dataset, tmin, tmax, exp_time_periods[:2], savepath
            )

        # Pre-processing and return power of Hilbert Transform.
        print("Applying pre-processing...")
        if pipe == "simple":
            epochs, labels, _, _, _ = apply_preprocessing(
                epochs,
                labels,
                meta,
                channels,
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )

            # Filtering with channel-specific custom frequency band.
            for cb, ch in zip(power_band, channels):
                epochs.filter(cb[0], cb[1], picks=ch)
            # Power.
            epochs.apply_hilbert(envelope=True)
            epochs = epochs.get_data()
            epochs = epochs**2

        elif pipe == "csp":
            epochs, labels, _, _, _ = apply_preprocessing(
                epochs,
                labels,
                meta,
                channels,
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )

            # Filtering with channel-specific custom frequency band.
            for cb, ch in zip(power_band, channels):
                epochs.filter(cb[0], cb[1], picks=ch)
            epochs = epochs.get_data()

        # Trim time to match task duration and keep selected channels.
        epochs = epochs[:, :, toi]
        epochs = epochs[:, channel_ids, :]
        all_sfl = np.arange(0, epochs.shape[0], 1)

        # Repeated cross-validation.
        for rep in range(reps):
            # Trial shuffling.
            np.random.shuffle(all_sfl)
            features = np.copy(epochs)[all_sfl, :]
            tt_labels = np.copy(labels)[all_sfl]

            if pipe == "simple":
                if metric == "accuracy":
                    mean_accs[s, rep, :] = (
                        cross_val_score(
                            pipeline,
                            np.mean(features, axis=-1),
                            tt_labels,
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )
                        * 100
                    )
                elif metric == "rocauc":
                    mean_accs[s, rep, :] = cross_val_score(
                        pipeline,
                        np.mean(features, axis=-1),
                        tt_labels,
                        scoring="roc_auc",
                        cv=n_folds,
                        n_jobs=n_jobs,
                    )
            elif pipe == "csp":
                if metric == "accuracy":
                    mean_accs[s, rep, :] = (
                        cross_val_score(
                            pipeline, features, tt_labels, cv=n_folds, n_jobs=n_jobs
                        )
                        * 100
                    )
                elif metric == "rocauc":
                    mean_accs[s, rep, :] = cross_val_score(
                        pipeline,
                        features,
                        tt_labels,
                        scoring="roc_auc",
                        cv=n_folds,
                        n_jobs=n_jobs,
                    )

        print("\n")

    # Save results.
    std_accs = np.std(mean_accs, axis=(1, 2))
    np.save(
        join(
            savepath,
            "mean_{}_power_{}_band_{}{}".format(metric, band, pipe, fooof_save_str),
        ),
        mean_accs,
    )
    np.save(
        join(
            savepath,
            "std_{}_power_{}_band_{}{}".format(metric, band, pipe, fooof_save_str),
        ),
        std_accs,
    )


def classify_tf_features(
    subjects,
    exp_variables,
    channel_ids,
    savepath,
    classifier,
    trials_fraction=0.2,
    band="beta",
    remove_fooof=True,
    metric="rocauc",
    stratification=5,
    reps=100,
    n_chars_tk=4,
    n_groups=9,
    n_folds=5,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based on burst
    characteristics features.

    Optionally, adopt a stratification approach for creating the validation set,
    in order to eliminate the randomness introduced in the results by the specific
    trials that have been used each time during this step.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    channel_ids: list
                 Indices of channels to take into account during burst
                 dictionary creation.
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    classifier: scikit-learn model
                Any scikit learn model that can be used as a classifier
                (e.g. LDA, SVC).
    trials_fraction: float, optional
                     Fraction of total trials in each subject's data
                     used to create the burst dictionary.
                     Defaults to 0.2.
    band: str {"mu", "beta"}, optional
          Select band for burst detection.
          Defaults to "beta".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    stratification: int, optional
                    Split the dictionary's trials in (almost) equal strata;
                    the assumption is that you are interested in a repeated
                    estimation of the dictionary and the dimensionality
                    reduction model.
                    Defaults to 5.
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    n_chars_tk: int, optional
                Maximum number of characteristics to explore.
                Defaults to 4 (the maximum).
    n_groups: int, optional
              Number of groups for the scores of each characteristic
              axis should be split into.
              Defaults to 9.
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # Variables for storing results initialization.
    mean_accs = np.empty([len(subjects), stratification])
    std_accs = np.empty([len(subjects)])

    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    for subject_id, subject in enumerate(subjects):
        print("Estimating decoding accuracy for subject {}...".format(subject))

        # Subject-specific training labels.
        # Suject-specific directory.
        labels = np.load(join(savepath, "sub_{}/labels.npy".format(subject)))

        # Burst space model.
        bspace = BurstSpace(
            exp_variables,
            subject,
            trials_fraction=trials_fraction,
            channel_ids=channel_ids,
            remove_fooof=remove_fooof,
            band=band,
        )

        subject_dictionary, sub_chars_dists = bspace.compute_chars_dists(subject)
        missing_trials = bspace.missing_trials
        exist_labels = np.delete(np.copy(labels), missing_trials)

        # Shuffle the data.
        all_sfl = np.arange(0, len(exist_labels), 1)
        np.random.shuffle(all_sfl)

        kf = StratifiedKFold(n_splits=stratification)
        for i, (traintest_ids, val_ids) in enumerate(
            kf.split(np.zeros(len(all_sfl)), exist_labels[all_sfl])
        ):
            if n_chars_tk == 1:
                mean_accs_hp = np.empty([n_groups - 1, n_chars_tk])
            else:
                mean_accs_hp = np.empty([n_groups - 1, n_chars_tk - 1])

            # Hyper-parameters exploration.
            # At least two features per component are required.
            for comp_groups in range(2, n_groups + 1):
                gid = comp_groups - 2

                # Separate exploration for burst volume and for the other characteristics.
                if n_chars_tk == 1:
                    cta = [1]
                else:
                    cta = [2]

                while True:
                    tfclf = TfFeatures(exp_variables)
                    bc_features, bc_labels = tfclf.transform(
                        subject_dictionary,
                        sub_chars_dists,
                        labels,
                        channel_ids,
                        cta,
                        n_groups,
                    )

                    tt_features = np.copy(bc_features)[all_sfl, :][traintest_ids, :]
                    tt_labels = np.copy(bc_labels)[all_sfl][traintest_ids]

                    # Initialize an array to store the results per repetition.
                    accuracies = np.zeros([reps, n_folds])

                    # Repeated cross-validation.
                    for r in range(reps):
                        r_ids = np.arange(0, len(tt_labels), 1)
                        np.random.shuffle(r_ids)
                        tt_features = tt_features[r_ids, :]
                        tt_labels = tt_labels[r_ids]

                        if metric == "accuracy":
                            accuracies[r, :] = (
                                cross_val_score(
                                    classifier,
                                    tt_features,
                                    tt_labels,
                                    cv=n_folds,
                                    n_jobs=n_jobs,
                                )
                                * 100
                            )
                        elif metric == "rocauc":
                            accuracies[r, :] = cross_val_score(
                                classifier,
                                tt_features,
                                tt_labels,
                                scoring="roc_auc",
                                cv=n_folds,
                            )

                    # Store intermediate results.
                    ma = np.mean(accuracies)
                    mean_accs_hp[gid, cta[-1] - 2] = ma

                    # Repeat up to the number of characteristics we are interested in.
                    if cta[-1] < n_chars_tk:
                        cta.append(cta[-1] + 1)
                    else:
                        break

            # Identification of optimal hyperparameters.
            ids_g, ids_pc = np.unravel_index(
                np.argmax(mean_accs_hp), (mean_accs_hp.shape[0], mean_accs_hp.shape[1])
            )

            # Fit optimal model.
            classifier.fit(
                np.copy(bc_features)[all_sfl, :][traintest_ids, :],
                np.copy(bc_labels)[all_sfl][traintest_ids],
            )

            # Validation score.
            val_features = np.copy(bc_features)[all_sfl, :][val_ids, :]
            val_labels = np.copy(bc_labels)[all_sfl][val_ids]
            mean_accs[subject_id, i] = roc_auc_score(
                val_labels, classifier.predict_proba(val_features)[:, 1]
            )
        std_accs[subject_id] = np.std(mean_accs[subject_id, :])

        print("\n")

    # Save results.
    if n_chars_tk == 1:
        np.save(
            join(
                savepath, "mean_{}_{}{}_tf_volume".format(metric, band, fooof_save_str)
            ),
            mean_accs,
        )
        np.save(
            join(
                savepath, "std_{}_{}{}_tf_volume".format(metric, band, fooof_save_str)
            ),
            std_accs,
        )
    elif n_chars_tk == 2:
        np.save(
            join(
                savepath,
                "mean_{}_{}{}_tf_amplitude".format(metric, band, fooof_save_str),
            ),
            mean_accs,
        )
        np.save(
            join(
                savepath,
                "std_{}_{}{}_tf_amplitude".format(metric, band, fooof_save_str),
            ),
            std_accs,
        )
    else:
        np.save(
            join(
                savepath,
                "mean_{}_{}{}_tf_features".format(metric, band, fooof_save_str),
            ),
            mean_accs,
        )
        np.save(
            join(
                savepath, "std_{}_{}{}_tf_features".format(metric, band, fooof_save_str)
            ),
            std_accs,
        )


def classify_burst_conv_power(
    subjects,
    dataset,
    dataset_name,
    exp_variables,
    channels,
    channel_ids,
    zapit,
    noise_freq,
    noise_wins,
    savepath,
    comps_to_analyze,
    comps_groups,
    kernel,
    pipe,
    pipeline,
    band="beta",
    tf_method="superlets",
    remove_fooof=False,
    stratification=None,
    trials_fraction=0.2,
    solver="pca",
    model_type="dataset",
    winsorization=[2, 98],
    output_waveforms="extrema",
    n_comps=3,
    metric="rocauc",
    reps=100,
    classification_mode="trial",
    sl_win=[1.0, 0.05],
    time_res_lims=[-0.5, 0.5],
    n_folds=5,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based on frequency
    band power.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    dataset: MOABB object
             Dataset from the MOABB prject for the analysis.
    dataset_name: str
                  Corresponding ddtaset name.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    channels: list
              Names of channels to keep while pre-processing the 'dataset'.
    channel_ids: list
                 Indices of channels to keep for the analysis.
    zapit: bool
           If set to "True", iteratively remove a noise artifact from the raw
           signal. The frequency of the artifact is provided by 'this_freq'.
    noise_freq: int or None
               When set to "int", frequency containing power line noise, or
               equivalent artifact. Only considered if 'zapit' is "True".
    noise_wins: list or None
                Window sizes for removing line noise.  Only considered if
                'zapit' is "True".
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    comps_to_analyze: list or numpy array
                      List of the indices of components used for
                      feature creation.
    comps_groups: int
                  Number of groups the scores of each component
                  axis should be split into.
    kernel: str {"drm_components", "waveforms"}
            Select the kernel for performing the convolution with the original
            signal. If set to "drm_components" convloution uses the dimenionality
            reduction components. If set to "waveforms" convolution uses the
            corresponding 'output_waveforms'.
    pipe: str {"simple", "csp"}
          Naming convention that reflects 'pipeline' for storing results.
    pipeline: scikit-learn model
              Any scikit learn pipeline of models.
              (e.g. CSP & LDA, SVC).
    band: str {"beta", "mu", "mu_beta"}, optional
              Frequency band for loading appropriate burst data.
              Defaults to "beta".
    tf_method: str {"wavelets", "superlets"}, optional
               String indicating the algorithm used for burst
               extraction.
               Defaults to "superlets".
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    stratification: int, str or None, optional
                    If set to 'int', split the dictionary's trials in (almost)
                    equal strata; the assumption is that you are interested in
                    a repeated estimation of the dictionary and the dimensionality
                    reduction model. If 'str', try to match it to an experimental
                    session. If None use a random sample of all trials according
                    to "trials_fraction".
                    Defaults to "None".
    trials_fraction: float, optional
                     Fraction of total trials in each subject's data
                     used to create the burst dictionary.
                     Defaults to 0.2.
    solver: str {"pca", "csp"}, optional
            Dimensionality reduction algorithm. Implements the PCA sklearn model,
            or the MNE-python CSP model.
            Defaults to "pca".
    model_type: str {"dataset", "subject"}, optional
                Option to build a dataset- or subject-specific dimensionality
                reduction model.
                Defaults to "dataset".
    winsorization: None or two-elements list, optional
                   Option to clip bursts with exreme values along a given
                   component axis to certain limits of the
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
    n_comps: int, optional
             Number of components ultimately kept among those indicated by
             'comps_to_analyze', based on the modulation index. This index indicates
             the relative difference between ipsilateral and contralateral average
             waveform modulation during the task period relative to baseline for each
             component.
             Defaults to 3.
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    classification_mode: str {"trial", "incremental", "sliding"}, optional
                         Choice of time-resolved or full trial decoding scheme.
                         Defaults to "trial".
    sl_win: two element list or numpy array, optional
            Time (in s) of the size of sliding windows and step between consecutive
            windows. Only used if 'classification_mode' is set to "sliding".
            Defaults to [1.0, 0.05]
    time_res_lims: two element list or numpy array, optional
                   Time (in s) before the beginning and ending of the task period,
                   for performing time-resolved decoding.
                   Defaults to [-0.5, 0.5]
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # ----- #
    # Time period of task.
    tmin = exp_variables["tmin"]
    tmax = exp_variables["tmax"]
    exp_time_periods = exp_variables["exp_time_periods"]
    sfreq = exp_variables["sfreq"]

    exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
    exp_time = np.around(exp_time, decimals=3)
    task_time_lims = [exp_time_periods[1], exp_time_periods[2]]

    # Initialization of variables for storing results.
    if classification_mode == "trial":
        time_res_str = "_trial"

        mean_accs = np.zeros([len(subjects), reps, n_folds])
        std_accs = np.zeros([len(subjects)])

        # For consistency purposes only.
        n_windows = 1

    elif classification_mode == "incremental":
        time_res_str = "_tr"

        # Time windows.
        window_length = exp_variables["bin_dt"]
        samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
        window_samples = int(window_length / samples_step)

        baseline_window_end = np.where(
            exp_time <= task_time_lims[0] + time_res_lims[0]
        )[0][-1]
        rebound_window_end = np.where(exp_time <= task_time_lims[1] + time_res_lims[1])[
            0
        ][-1]
        n_windows = int((rebound_window_end - baseline_window_end) / window_samples) + 1

        mean_accs = np.zeros([len(subjects), n_windows, reps, n_folds])
        std_accs = np.zeros([len(subjects), n_windows])

    elif classification_mode == "sliding":
        time_res_str = "_sl"

        # Time windows.
        window_length = sl_win[0]
        step_length = sl_win[1]
        samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
        window_samples = window_length / samples_step
        step_samples = step_length / samples_step

        baseline_window_end = np.where(
            exp_time <= task_time_lims[0] + time_res_lims[0]
        )[0][-1]
        rebound_window_end = np.where(exp_time <= task_time_lims[1] + time_res_lims[1])[
            0
        ][-1]
        n_windows = int(
            np.floor(
                ((rebound_window_end - baseline_window_end) - window_samples)
                / step_samples
            )
            + window_samples / step_samples
            + 1
        )

        mean_accs = np.zeros([len(subjects), n_windows, reps, n_folds])
        std_accs = np.zeros([len(subjects), n_windows])

    # Naming convention strings.
    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    if isinstance(stratification, str):
        session_str = "_sess_{}".format(stratification)
    else:
        session_str = ""

    if kernel == "drm_components":
        conv_kernels_str = "comps"
    elif kernel == "waveforms":
        conv_kernels_str = "waves"

    # ----- #
    for s, subject in enumerate(subjects):
        print("Estimating decoding accuracy for subject {}...".format(subject))

        # Suject-specific directory.
        sub_dir = join(savepath, "sub_{}/".format(subject))

        # Subject's raw data loading.
        print("Loading raw data...")

        if dataset_name == "weibo2014":
            epochs, labels, meta = load_sub(
                subject,
                dataset,
                tmin,
                tmax,
                exp_time_periods[:2],
                savepath,
                band_pass=[0, 90],
            )
        else:
            epochs, labels, meta = load_sub(
                subject, dataset, tmin, tmax, exp_time_periods[:2], savepath
            )

        # Pre-processing.
        print("Applying pre-processing...")

        # Correspondence of trials used for burst detection and
        # trials used for classification.
        if channels == None:
            n_trials = len(labels)
            init_trials = np.arange(0, n_trials, 1)

            if (
                dataset_name == "munichmi"
                or dataset_name == "cho2017"
                or dataset_name == "dreyer2023"
            ):
                dummy_epochs, _, _, _, _ = apply_preprocessing(
                    epochs.copy(),
                    labels,
                    meta,
                    channels=exp_variables["channels"],
                    zapit=True,
                    noise_freq=noise_freq,
                    noise_wins=noise_wins,
                    return_epochs=True,
                )
            else:
                dummy_epochs, _, _, _, _ = apply_preprocessing(
                    epochs.copy(),
                    labels,
                    meta,
                    channels=exp_variables["channels"],
                    zapit=zapit,
                    noise_freq=noise_freq,
                    noise_wins=noise_wins,
                    return_epochs=True,
                )

            epochs, labels, _, _, _ = apply_preprocessing(
                epochs,
                labels,
                meta,
                channels=channels,
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )

            dummy_drop = []
            for tr, ep in enumerate(dummy_epochs.drop_log):
                if len(ep) != 0:
                    dummy_drop.append(tr)
            dummy_tr_kept = np.delete(init_trials, dummy_drop)

            ep_drop = []
            for tr, ep in enumerate(epochs.drop_log):
                if len(ep) != 0:
                    ep_drop.append(tr)
            tr_kept = np.delete(init_trials, ep_drop)

        else:
            epochs, labels, _, _, _ = apply_preprocessing(
                epochs,
                labels,
                meta,
                channels=channels,
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )

        # ----- #
        # Burst space model.
        bspace = BurstSpace(
            exp_variables,
            subjects if model_type == "dataset" else subject,
            trials_fraction=trials_fraction,
            channel_ids=channel_ids,
            remove_fooof=remove_fooof,
            band=band,
            stratification=stratification,
            verbose=False,
        )
        bspace.fit_transform(
            solver=solver, n_components=comps_to_analyze[-1], output="waveforms"
        )
        drm_components, binned_waveforms, _, _ = bspace.estimate_waveforms(
            comps_to_analyze,
            comps_groups,
            winsorization=winsorization,
            output_waveforms=output_waveforms,
            n_comps=n_comps,
        )
        drm_trials = bspace.drm_trials[s] if model_type == "dataset" else bspace.drm_trials[0]

        # Kernel selection and convolution.
        if kernel == "drm_components":
            conv_kernels = drm_components
        elif kernel == "waveforms":
            conv_kernels = []
            for comp_axis in binned_waveforms:
                for binned_waveform in comp_axis:
                    conv_kernels.append(binned_waveform)

        # If needed, find correspondence between trials in burst dictionary
        # and classification trials.
        if channels == None and len(drm_trials) != 0:
            drm_trs = np.intersect1d(tr_kept, dummy_tr_kept[drm_trials])
            temp = []
            for k in drm_trs:
                if k in tr_kept:
                    temp.append(int(np.where(tr_kept == k)[0]))

            drm_trials = temp

        # ----- #
        # Optional channel selection.
        orig_data = epochs.copy().get_data()
        if pipe == "simple":
            orig_data = orig_data[:, channel_ids, :]

        # Convolution with selected kernels.
        conv_data = []
        for conv_kernel in conv_kernels:
            # Remove trials whose data were used while creating the burst space model.
            conv_kernel_data = np.delete(np.copy(orig_data), drm_trials, axis=0)

            conv_kernel_data = np.apply_along_axis(
                np.convolve, -1, conv_kernel_data, conv_kernel, mode="same"
            )

            conv_data.append(conv_kernel_data)

            n_trials = conv_kernel_data.shape[0]

        # ----- #
        # Time resolution.
        for n in range(n_windows):
            if classification_mode == "trial":
                toi = np.where(
                    np.logical_and(
                        exp_time >= task_time_lims[0], exp_time <= task_time_lims[1]
                    )
                )[0]

            elif classification_mode == "incremental":
                window_end = exp_time[baseline_window_end + n * window_samples]
                if window_end <= exp_time_periods[1]:
                    # Baseline activity.
                    toi = np.where(exp_time <= window_end)[0]
                else:
                    # Do not take into account baseline activity.
                    toi = np.where(
                        np.logical_and(
                            exp_time >= task_time_lims[0], exp_time <= window_end
                        )
                    )[0]
                print("\tTime window: {} s".format(window_end))

            elif classification_mode == "sliding":
                window_start = exp_time[int(n * step_samples)]
                window_end = exp_time[int(n * step_samples + window_samples)]
                toi = np.where(
                    np.logical_and(exp_time >= window_start, exp_time <= window_end)
                )[0]
                print("\tTime window: {} to {} s".format(window_start, window_end))

            # Time window selection.
            conv_data_toi = []
            for conv_kernel_data in conv_data:
                # Trim time.
                conv_kernel_data = conv_kernel_data[:, :, toi]

                # Amplitude envelope.
                if pipe == "simple":
                    conv_kernel_data = conv_kernel_data**2
                    conv_kernel_data = np.mean(conv_kernel_data, axis=-1)

                conv_data_toi.append(conv_kernel_data)

            # ----- #
            # Repeated cross-validation.
            for rep in range(reps):
                # Classification.
                all_sfl = np.arange(0, n_trials, 1)
                t_labels = np.delete(np.copy(labels), drm_trials)

                if pipe == "simple":
                    if len(conv_data) > 1:
                        all_features = np.hstack(np.copy(conv_data_toi))
                    else:
                        all_features = np.copy(conv_data_toi[0])

                    # Trial shuffling.
                    np.random.shuffle(all_sfl)
                    all_features = all_features[all_sfl, :]
                    tt_labels = np.copy(t_labels)[all_sfl]

                    if classification_mode == "trial" and metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline,
                                all_features,
                                tt_labels,
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif classification_mode == "trial" and metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline,
                            all_features,
                            tt_labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "accuracy":
                        mean_accs[s, n, rep, :] = (
                            cross_val_score(
                                pipeline,
                                all_features,
                                tt_labels,
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "rocauc":
                        mean_accs[s, n, rep, :] = cross_val_score(
                            pipeline,
                            all_features,
                            tt_labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )

                elif pipe == "csp":
                    # Trial shuffling.
                    rs = StratifiedKFold(n_splits=n_folds, shuffle=True)

                    for i, (train_ids, test_ids) in enumerate(
                        rs.split(all_sfl, t_labels)
                    ):
                        # CSP transformation per band.
                        train_epochs = []
                        test_epochs = []
                        train_labels = np.copy(t_labels)[train_ids]
                        test_labels = np.copy(t_labels)[test_ids]
                        for epoch in conv_data_toi:
                            train_epoch = pipeline[0].fit_transform(
                                epoch[train_ids], train_labels
                            )
                            train_epochs.append(train_epoch)
                            test_epoch = pipeline[0].transform(epoch[test_ids])
                            test_epochs.append(test_epoch)

                        if len(conv_data_toi) > 1:
                            train_features = np.hstack(train_epochs)
                            test_features = np.hstack(test_epochs)
                        else:
                            train_features = train_epochs[0]
                            test_features = test_epochs[0]

                        # Classifier training on all CSP features.
                        pipeline[1].fit(train_features, train_labels)

                        # Classifier testing on all CSP features.
                        if classification_mode == "trial" and metric == "accuracy":
                            mean_accs[s, rep, i] = (
                                pipeline[1].score(test_features, test_labels) * 100
                            )
                        elif classification_mode == "trial" and metric == "rocauc":
                            mean_accs[s, rep, i] = roc_auc_score(
                                test_labels,
                                pipeline[1].predict_proba(test_features)[:, 1],
                            )
                        elif (
                            classification_mode == "incremental"
                            or classification_mode == "sliding"
                        ) and metric == "accuracy":
                            mean_accs[s, n, rep, i] = (
                                pipeline[1].score(test_features, test_labels) * 100
                            )
                        elif (
                            classification_mode == "incremental"
                            or classification_mode == "sliding"
                        ) and metric == "rocauc":
                            mean_accs[s, n, rep, i] = roc_auc_score(
                                test_labels,
                                pipeline[1].predict_proba(test_features)[:, 1],
                            )

                elif pipe == "riemann":
                    # Trial shuffling.
                    rs = StratifiedKFold(n_splits=n_folds, shuffle=True)

                    train_cov = []
                    for cnvd in conv_data_toi:
                        train_cov.append(
                            pipeline[2].fit_transform(
                                pipeline[1].fit_transform(pipeline[0].transform(cnvd))
                            )
                        )
                    train_cov = np.hstack(train_cov)

                    if classification_mode == "trial" and metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline[-1],
                                train_cov,
                                t_labels,
                                cv=rs,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif classification_mode == "trial" and metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline[-1],
                            train_cov,
                            t_labels,
                            scoring="roc_auc",
                            cv=rs,
                            n_jobs=n_jobs,
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "accuracy":
                        mean_accs[s, n, rep, :] = (
                            cross_val_score(
                                pipeline[-1],
                                train_cov,
                                t_labels,
                                cv=rs,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "rocauc":
                        mean_accs[s, n, rep, :] = cross_val_score(
                            pipeline[-1],
                            train_cov,
                            t_labels,
                            scoring="roc_auc",
                            cv=rs,
                            n_jobs=n_jobs,
                        )

            if (
                classification_mode == "incremental"
                or classification_mode == "sliding"
            ):
                std_accs[s, n] = np.std(mean_accs[s, n, :, :])

        print("\n")

    # Save results.
    if classification_mode == "trial":
        std_accs = np.std(mean_accs, axis=(1, 2))

    np.save(
        join(
            savepath,
            "mean_{}_power_{}_band_{}{}_conv_{}{}{}".format(
                metric,
                band,
                pipe,
                fooof_save_str,
                conv_kernels_str,
                time_res_str,
                session_str,
            ),
        ),
        mean_accs,
    )
    np.save(
        join(
            savepath,
            "std_{}_power_{}_band_{}{}_conv_{}{}{}".format(
                metric,
                band,
                pipe,
                fooof_save_str,
                conv_kernels_str,
                time_res_str,
                session_str,
            ),
        ),
        std_accs,
    )


def classify_fb_power(
    subjects,
    dataset,
    dataset_name,
    exp_variables,
    filter_bank,
    channels,
    channel_ids,
    zapit,
    noise_freq,
    noise_wins,
    savepath,
    pipe,
    pipeline,
    metric="rocauc",
    reps=100,
    classification_mode="trial",
    sl_win=[1.0, 0.05],
    time_res_lims=[-0.5, 0.5],
    n_folds=5,
    n_jobs=-1,
):
    """
    Perform a repeated cross-validation for classification based on frequency
    band power.

    Parameters
    ----------
    subjects: list
              Subjects to be analyzed.
    dataset: MOABB object
             Dataset from the MOABB prject for the analysis.
    dataset_name: str
                  Corresponding ddtaset name.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    filter_bank: list
                 List of two-element lists or numpy arrays that define the
                 bands for performing a filter bank analysis.
    channels: list
              Names of channels to keep while pre-processing the 'dataset'.
    channel_ids: list
                 Indices of channels to keep for the analysis.
    zapit: bool
           If set to "True", iteratively remove a noise artifact from the raw
           signal. The frequency of the artifact is provided by 'this_freq'.
    noise_freq: int or None
               When set to "int", frequency containing power line noise, or
               equivalent artifact. Only considered if 'zapit' is "True".
    noise_wins: list or None
                Window sizes for removing line noise.  Only considered if
                'zapit' is "True".
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    pipe: str {"simple", "csp", "riemann"}
          Naming convention that reflects 'pipeline' for storing results.
    pipeline: scikit-learn model
              Any scikit learn pipeline of models.
              (e.g. CSP & LDA, SVC).
    metric: str, {"rocauc", "accuracy"}, optional
            Metric for estimating the classification score.
            Defaults to "rocauc".
    reps: int, optional
          Number of repetitions for shuffling the order of the data, and
          estimating the classification score.
          Deafualts to 100.
    classification_mode: str {"trial", "incremental", "sliding"}, optional
                         Choice of time-resolved or full trial decoding scheme.
                         Defaults to "trial".
    sl_win: two element list or numpy array, optional
            Time (in s) of the size of sliding windows and step between consecutive
            windows. Only used if 'classification_mode' is set to "sliding".
            Defaults to [1.0, 0.05]
    time_res_lims: two element list or numpy array, optional
                   Time (in s) before the beginning and ending of the task period,
                   for performing time-resolved decoding.
                   Defaults to [-0.5, 0.5]
    n_folds: int, optional
             Number of folds for cross-validation.
             Defaults to 5.
    n_jobs: int, optional
            Number of jobs according to scikit standards.
            Defaults to -1.
    """

    # ----- #
    # Time period of task.
    tmin = exp_variables["tmin"]
    tmax = exp_variables["tmax"]
    exp_time_periods = exp_variables["exp_time_periods"]
    sfreq = exp_variables["sfreq"]

    exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
    exp_time = np.around(exp_time, decimals=3)
    task_time_lims = [exp_time_periods[1], exp_time_periods[2]]

    # Initialization of variables for storing results.
    if classification_mode == "trial":
        time_res_str = "_trial"

        mean_accs = np.zeros([len(subjects), reps, n_folds])
        std_accs = np.zeros([len(subjects)])

        # For consistency purposes only.
        n_windows = 1

    elif classification_mode == "incremental":
        time_res_str = "_tr"

        # Time windows.
        window_length = exp_variables["bin_dt"]
        samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
        window_samples = int(window_length / samples_step)

        baseline_window_end = np.where(
            exp_time <= task_time_lims[0] + time_res_lims[0]
        )[0][-1]
        rebound_window_end = np.where(exp_time <= task_time_lims[1] + time_res_lims[1])[
            0
        ][-1]
        n_windows = int((rebound_window_end - baseline_window_end) / window_samples) + 1

        mean_accs = np.zeros([len(subjects), n_windows, reps, n_folds])
        std_accs = np.zeros([len(subjects), n_windows])

    elif classification_mode == "sliding":
        time_res_str = "_sl"

        # Time windows.
        window_length = sl_win[0]
        step_length = sl_win[1]
        samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
        window_samples = window_length / samples_step
        step_samples = step_length / samples_step

        baseline_window_end = np.where(
            exp_time <= task_time_lims[0] + time_res_lims[0]
        )[0][-1]
        rebound_window_end = np.where(exp_time <= task_time_lims[1] + time_res_lims[1])[
            0
        ][-1]
        n_windows = int(
            np.floor(
                ((rebound_window_end - baseline_window_end) - window_samples)
                / step_samples
            )
            + window_samples / step_samples
            + 1
        )

        mean_accs = np.zeros([len(subjects), n_windows, reps, n_folds])
        std_accs = np.zeros([len(subjects), n_windows])

    # ----- #
    for s, subject in enumerate(subjects):
        print("Estimating decoding accuracy for subject {}...".format(subject))

        # Suject-specific directory.
        sub_dir = join(savepath, "sub_{}/".format(subject))

        # Subject's raw data loading.
        print("Loading raw data...")

        if dataset_name == "weibo2014":
            epochs, labels, meta = load_sub(
                subject,
                dataset,
                tmin,
                tmax,
                exp_time_periods[:2],
                savepath,
                band_pass=[0, 90],
            )
        else:
            epochs, labels, meta = load_sub(
                subject, dataset, tmin, tmax, exp_time_periods[:2], savepath
            )

        # Pre-processing.
        print("Applying pre-processing...")

        epochs, labels, _, _, _ = apply_preprocessing(
            epochs,
            labels,
            meta,
            channels,
            zapit=zapit,
            noise_freq=noise_freq,
            noise_wins=noise_wins,
            return_epochs=True,
        )

        # ----- #
        # Filtering with channel-specific custom frequency bands.
        all_epochs = []
        for fb in filter_bank:
            # Filtering.
            filter_band_epochs = epochs.copy().filter(fb[0], fb[1])

            if pipe == "simple":
                # Hilbert Power.
                filter_band_epochs.apply_hilbert(envelope=True)
                filter_band_epochs = filter_band_epochs.get_data()
                filter_band_epochs = filter_band_epochs**2

            else:
                # Band-passed signals.
                filter_band_epochs = filter_band_epochs.get_data()
                if channels != None:
                    filter_band_epochs = filter_band_epochs[:, channel_ids, :]

            all_epochs.append(filter_band_epochs)

            n_trials = filter_band_epochs.shape[0]

        all_sfl = np.arange(0, n_trials, 1)

        # ----- #
        # Time resolution.
        for n in range(n_windows):
            if classification_mode == "trial":
                toi = np.where(
                    np.logical_and(
                        exp_time >= task_time_lims[0], exp_time <= task_time_lims[1]
                    )
                )[0]

            elif classification_mode == "incremental":
                window_end = exp_time[baseline_window_end + n * window_samples]
                if window_end <= exp_time_periods[1]:
                    # Baseline activity.
                    toi = np.where(exp_time <= window_end)[0]
                else:
                    # Do not take into account baseline activity.
                    toi = np.where(
                        np.logical_and(
                            exp_time >= task_time_lims[0], exp_time <= window_end
                        )
                    )[0]
                print("\tTime window: {} s".format(window_end))

            elif classification_mode == "sliding":
                window_start = exp_time[int(n * step_samples)]
                window_end = exp_time[int(n * step_samples + window_samples)]
                toi = np.where(
                    np.logical_and(exp_time >= window_start, exp_time <= window_end)
                )[0]
                print("\tTime window: {} to {} s".format(window_start, window_end))

            # Time window selection.
            all_epochs_toi = []
            for filter_band_epochs in all_epochs:
                # Trim time.
                filter_band_epochs = filter_band_epochs[:, :, toi]

                if pipe == "simple":
                    # Keep selected channels.
                    if channels != None:
                        filter_band_epochs = np.mean(
                            filter_band_epochs[:, channel_ids, :], axis=-1
                        )
                    else:
                        filter_band_epochs = np.mean(filter_band_epochs, axis=-1)

                all_epochs_toi.append(filter_band_epochs)

            # ----- #
            # Repeated cross-validation.
            for rep in range(reps):
                if pipe == "simple":
                    if len(all_epochs) > 1:
                        all_features = np.hstack(np.copy(all_epochs_toi))
                    else:
                        all_features = np.copy(all_epochs_toi[0])

                    # Trial shuffling.
                    np.random.shuffle(all_sfl)
                    all_features = all_features[all_sfl, :]
                    tt_labels = np.copy(labels)[all_sfl]

                    if classification_mode == "trial" and metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline,
                                all_features,
                                tt_labels,
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif classification_mode == "trial" and metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline,
                            all_features,
                            tt_labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "accuracy":
                        mean_accs[s, n, rep, :] = (
                            cross_val_score(
                                pipeline,
                                all_features,
                                tt_labels,
                                cv=n_folds,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "rocauc":
                        mean_accs[s, n, rep, :] = cross_val_score(
                            pipeline,
                            all_features,
                            tt_labels,
                            scoring="roc_auc",
                            cv=n_folds,
                            n_jobs=n_jobs,
                        )

                elif pipe == "csp":
                    # Trial shuffling.
                    rs = StratifiedKFold(n_splits=n_folds, shuffle=True)

                    for i, (train_ids, test_ids) in enumerate(
                        rs.split(all_sfl, np.copy(labels))
                    ):
                        # CSP transformation per band.
                        train_epochs = []
                        test_epochs = []
                        train_labels = np.copy(labels)[train_ids]
                        test_labels = np.copy(labels)[test_ids]
                        for epoch in all_epochs_toi:
                            train_epoch = pipeline[0].fit_transform(
                                epoch[train_ids], train_labels
                            )
                            train_epochs.append(train_epoch)
                            test_epoch = pipeline[0].transform(epoch[test_ids])
                            test_epochs.append(test_epoch)

                        if len(all_epochs_toi) > 1:
                            train_features = np.hstack(train_epochs)
                            test_features = np.hstack(test_epochs)
                        else:
                            train_features = train_epochs[0]
                            test_features = test_epochs[0]

                        # Classifier training on all CSP features.
                        pipeline[1].fit(train_features, train_labels)

                        # Classifier testing on all CSP features.
                        if classification_mode == "trial" and metric == "accuracy":
                            mean_accs[s, rep, i] = (
                                pipeline[1].score(test_features, test_labels) * 100
                            )
                        elif classification_mode == "trial" and metric == "rocauc":
                            mean_accs[s, rep, i] = roc_auc_score(
                                test_labels,
                                pipeline[1].predict_proba(test_features)[:, 1],
                            )
                        elif (
                            classification_mode == "incremental"
                            or classification_mode == "sliding"
                        ) and metric == "accuracy":
                            mean_accs[s, n, rep, i] = (
                                pipeline[1].score(test_features, test_labels) * 100
                            )
                        elif (
                            classification_mode == "incremental"
                            or classification_mode == "sliding"
                        ) and metric == "rocauc":
                            mean_accs[s, n, rep, i] = roc_auc_score(
                                test_labels,
                                pipeline[1].predict_proba(test_features)[:, 1],
                            )

                elif pipe == "riemann":
                    # Trial shuffling.
                    rs = StratifiedKFold(n_splits=n_folds, shuffle=True)

                    train_cov = []
                    for ep_toi in all_epochs_toi:
                        train_cov.append(
                            pipeline[2].fit_transform(
                                pipeline[1].fit_transform(pipeline[0].transform(ep_toi))
                            )
                        )
                    train_cov = np.hstack(train_cov)

                    if classification_mode == "trial" and metric == "accuracy":
                        mean_accs[s, rep, :] = (
                            cross_val_score(
                                pipeline[-1],
                                train_cov,
                                labels,
                                cv=rs,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif classification_mode == "trial" and metric == "rocauc":
                        mean_accs[s, rep, :] = cross_val_score(
                            pipeline[-1],
                            train_cov,
                            labels,
                            scoring="roc_auc",
                            cv=rs,
                            n_jobs=n_jobs,
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "accuracy":
                        mean_accs[s, n, rep, :] = (
                            cross_val_score(
                                pipeline[-1],
                                train_cov,
                                labels,
                                cv=rs,
                                n_jobs=n_jobs,
                            )
                            * 100
                        )
                    elif (
                        classification_mode == "incremental"
                        or classification_mode == "sliding"
                    ) and metric == "rocauc":
                        mean_accs[s, n, rep, :] = cross_val_score(
                            pipeline[-1],
                            train_cov,
                            labels,
                            scoring="roc_auc",
                            cv=rs,
                            n_jobs=n_jobs,
                        )

            if (
                classification_mode == "incremental"
                or classification_mode == "sliding"
            ):
                std_accs[s, n] = np.std(mean_accs[s, n, :, :])

        print("\n")

    # Save results.
    if classification_mode == "trial":
        std_accs = np.std(mean_accs, axis=(1, 2))

    np.save(
        join(
            savepath,
            "mean_{}_power_filter_bank_{}_{}_{}_{}{}".format(
                metric,
                len(filter_bank),
                pipe,
                filter_bank[0][0],
                filter_bank[-1][-1],
                time_res_str,
            ),
        ),
        mean_accs,
    )
    np.save(
        join(
            savepath,
            "std_{}_power_filter_bank_{}_{}_{}_{}{}".format(
                metric,
                len(filter_bank),
                pipe,
                filter_bank[0][0],
                filter_bank[-1][-1],
                time_res_str,
            ),
        ),
        std_accs,
    )
