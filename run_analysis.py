"""
Beta-burst analysis pipeline for motor classification.

Compares beta-burst detection vs traditional beta-filtering approaches
for classifying motor intention in healthy subjects and patients.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from mne.decoding import CSP
from mne.filter import filter_data


def run_analysis(subject_type, analysis_type):
    """
    Run beta-burst or beta-filter classification analysis.
    
    Returns accuracy, AUC scores, and statistics for each subject.
    """
    basepath = f"/mnt/data/Divyanshi/datasets_to_work/Perrine/Preprocessed/Notch_45/decim_4/{subject_type}_data"
    
    # Analysis parameters
    subsample_fraction = 0.2           # Fraction of trials for burst dictionary
    n_pca_components = 10              # PCA dimensionality reduction
    n_bins = 7                         # Bins for PCA component binning
    
    # Time window for burst extraction (seconds)
    TIME_WINDOW_START = 2.0
    TIME_WINDOW_END = 10.0
    
    # Classification parameters
    N_CSP_COMPONENTS = 4               # CSP components for spatial filtering
    N_RUNS = 5                         # Number of CV repetitions
    N_CV_SPLITS = 2                    # Cross-validation folds
    SAMPLING_FREQ = 250                # Sampling frequency (Hz)
    
    # Frequency bands for beta analysis (Hz)
    BETA_BANDS = {
        'band_1': (15, 17),
        'band_2': (17, 19),
        'band_3': (19, 22),
        'band_4': (22, 24),
        'band_5': (24, 26),
        'band_6': (26, 30),
    }
    
    # Initialize storage lists
    burst_list_waveform_c3_move = []
    burst_list_waveform_c5_move = []
    peak_times_move = []
    burst_list_waveform_c3_rest = []
    burst_list_waveform_c5_rest = []
    peak_times_rest = []
    
    # Determine subject range
    if subject_type == "Patient":
        subject_range = [i for i in range(1, 55) if i not in [9, 10, 16, 17, 18]]
    elif subject_type == "Subject":
        subject_range = range(1, 31)
    else:
        raise ValueError(f"Unknown subject type: {subject_type}")
    
    # Collect bursts from multiple subjects
    all_sub_waveforms = []
    
    for subject in subject_range:
        subject = f"{subject:02d}"
        sub_dir_movement = os.path.join(basepath, "movement", f"sub_{subject}")
        sub_dir_rest = os.path.join(basepath, "rest", f"sub_{subject}")
        
        file_name = "beta_bursts_superlets_nfs.npy"
        file_path_movement = os.path.join(sub_dir_movement, file_name)
        file_path_rest = os.path.join(sub_dir_rest, file_name)
        
        # Check if burst files exist before loading
        if not os.path.exists(file_path_movement):
            print(f"Warning: Missing movement burst file for subject {subject}, skipping")
            continue
        if not os.path.exists(file_path_rest):
            print(f"Warning: Missing rest burst file for subject {subject}, skipping")
            continue
        
        data_move = np.load(file_path_movement, allow_pickle=True)
        data_rest = np.load(file_path_rest, allow_pickle=True)
        
        # Select common trials between channels 3 and 5 for movement
        trial_c3_move = np.unique(data_move[3]['trial'])
        trial_c5_move = np.unique(data_move[5]['trial'])
        common_trials_move = np.intersect1d(trial_c3_move, trial_c5_move)
        n_sample_move = max(1, int(len(common_trials_move) * subsample_fraction))
        trial_random_move = np.random.choice(common_trials_move, size=n_sample_move, replace=False)
        
        c3_ids_move = []
        c5_ids_move = []
        for trial in trial_random_move:
            c3_ids_move.append(np.where(data_move[3]['trial'] == trial)[0])
            c5_ids_move.append(np.where(data_move[5]['trial'] == trial)[0])
        
        c3_ids_move = np.hstack(c3_ids_move)
        c5_ids_move = np.hstack(c5_ids_move)
        
        # Extract waveforms and peak times for channel 3
        waveform_c3_move = data_move[3]['waveform'][c3_ids_move]
        peak_c3_move = data_move[3]['peak_time'][c3_ids_move]
        
        # Filter by time window
        valid_mask = (peak_c3_move >= TIME_WINDOW_START) & (peak_c3_move <= TIME_WINDOW_END)
        waveform_c3_move = waveform_c3_move[valid_mask]
        peak_c3_move = peak_c3_move[valid_mask]
        
        # Extract waveforms and peak times for channel 5
        waveform_c5_move = data_move[5]['waveform'][c5_ids_move]
        peak_c5_move = data_move[5]['peak_time'][c5_ids_move]
        
        valid_mask = (peak_c5_move >= TIME_WINDOW_START) & (peak_c5_move <= TIME_WINDOW_END)
        waveform_c5_move = waveform_c5_move[valid_mask]
        peak_c5_move = peak_c5_move[valid_mask]
        
        # Save movement trial indices
        movement_indices = os.path.join(sub_dir_movement, 'indices')
        os.makedirs(movement_indices, exist_ok=True)
        np.savez(os.path.join(movement_indices, 'random_trial_indices.npz'), 
                 trial_random=trial_random_move)
        
        # Repeat for rest data
        trial_c3_rest = np.unique(data_rest[3]['trial'])
        trial_c5_rest = np.unique(data_rest[5]['trial'])
        common_trials_rest = np.intersect1d(trial_c3_rest, trial_c5_rest)
        n_sample_rest = max(1, int(len(common_trials_rest) * subsample_fraction))
        trial_random_rest = np.random.choice(common_trials_rest, size=n_sample_rest, replace=False)
        
        c3_ids_rest = []
        c5_ids_rest = []
        for trial in trial_random_rest:
            c3_ids_rest.append(np.where(data_rest[3]['trial'] == trial)[0])
            c5_ids_rest.append(np.where(data_rest[5]['trial'] == trial)[0])
        
        c3_ids_rest = np.hstack(c3_ids_rest)
        c5_ids_rest = np.hstack(c5_ids_rest)
        
        waveform_c3_rest = data_rest[3]['waveform'][c3_ids_rest]
        peak_c3_rest = data_rest[3]['peak_time'][c3_ids_rest]
        
        valid_mask = (peak_c3_rest >= TIME_WINDOW_START) & (peak_c3_rest <= TIME_WINDOW_END)
        waveform_c3_rest = waveform_c3_rest[valid_mask]
        peak_c3_rest = peak_c3_rest[valid_mask]
        
        waveform_c5_rest = data_rest[5]['waveform'][c5_ids_rest]
        peak_c5_rest = data_rest[5]['peak_time'][c5_ids_rest]
        
        valid_mask = (peak_c5_rest >= TIME_WINDOW_START) & (peak_c5_rest <= TIME_WINDOW_END)
        waveform_c5_rest = waveform_c5_rest[valid_mask]
        peak_c5_rest = peak_c5_rest[valid_mask]
        
        # Save rest trial indices
        rest_indices = os.path.join(sub_dir_rest, 'indices')
        os.makedirs(rest_indices, exist_ok=True)
        np.savez(os.path.join(rest_indices, 'random_trial_indices.npz'), 
                 trial_random=trial_random_rest)
        
        # Scale waveforms per subject
        scaler = RobustScaler()
        waveforms_move = np.vstack((waveform_c3_move, waveform_c5_move))
        waveforms_rest = np.vstack((waveform_c3_rest, waveform_c5_rest))
        waveforms_all = np.vstack((waveforms_move, waveforms_rest))
        waveforms_trans = scaler.fit_transform(waveforms_all)
        
        all_sub_waveforms.append(waveforms_trans)
        
        # Split back into separate lists
        burst_list_waveform_c3_move.append(waveforms_trans[:waveform_c3_move.shape[0], :])
        burst_list_waveform_c5_move.append(waveforms_trans[waveform_c3_move.shape[0]:waveform_c3_move.shape[0] + waveform_c5_move.shape[0], :])
        burst_list_waveform_c3_rest.append(waveforms_trans[waveform_c3_move.shape[0] + waveform_c5_move.shape[0]:waveform_c3_move.shape[0] + waveform_c5_move.shape[0] + waveform_c3_rest.shape[0], :])
        burst_list_waveform_c5_rest.append(waveforms_trans[waveform_c3_move.shape[0] + waveform_c5_move.shape[0] + waveform_c3_rest.shape[0]:, :])
        
        peak_times_move.append(peak_c3_move)
        peak_times_move.append(peak_c5_move)
        peak_times_rest.append(peak_c3_rest)
        peak_times_rest.append(peak_c5_rest)
    
    # Concatenate all bursts
    burst_array_waveform = np.vstack(all_sub_waveforms)
    peak_times_all = np.concatenate(peak_times_move + peak_times_rest)
    
    print(f"Total bursts shape: {burst_array_waveform.shape}")
    print(f"Total peak times shape: {peak_times_all.shape}")
    
    # Run PCA on burst waveforms
    pca = PCA(n_components=n_pca_components)
    pca_transformed = pca.fit_transform(burst_array_waveform)
    components_ = pca.components_
    
    # Clip extreme percentiles per PCA axis
    for pcax in range(n_pca_components):
        component_scores = pca_transformed[:, pcax]
        lower = np.percentile(component_scores, 2)
        upper = np.percentile(component_scores, 98)
        clipped_scores = np.clip(component_scores, lower, upper)
        pca_transformed[:, pcax] = clipped_scores
    
    # Concatenate class-specific burst arrays
    arr_c3_move = np.concatenate(burst_list_waveform_c3_move, axis=0)
    arr_c5_move = np.concatenate(burst_list_waveform_c5_move, axis=0)
    arr_c3_rest = np.concatenate(burst_list_waveform_c3_rest, axis=0)
    arr_c5_rest = np.concatenate(burst_list_waveform_c5_rest, axis=0)
    
    cond_labels = (
        ["c3_move"] * arr_c3_move.shape[0] +
        ["c5_move"] * arr_c5_move.shape[0] +
        ["c3_rest"] * arr_c3_rest.shape[0] +
        ["c5_rest"] * arr_c5_rest.shape[0]
    )
    
    waveform_times = data_move[3]['waveform_times']
    
    # Compute class means in PCA space
    n_c3_move = arr_c3_move.shape[0]
    n_c5_move = arr_c5_move.shape[0]
    n_c3_rest = arr_c3_rest.shape[0]
    n_c5_rest = arr_c5_rest.shape[0]
    
    proj_c3_move = pca_transformed[:n_c3_move]
    proj_c5_move = pca_transformed[n_c3_move:n_c3_move + n_c5_move]
    proj_c3_rest = pca_transformed[n_c3_move + n_c5_move:n_c3_move + n_c5_move + n_c3_rest]
    proj_c5_rest = pca_transformed[n_c3_move + n_c5_move + n_c3_rest:]
    
    avg_c3_move = np.mean(proj_c3_move, axis=0)
    avg_c5_move = np.mean(proj_c5_move, axis=0)
    avg_c3_rest = np.mean(proj_c3_rest, axis=0)
    avg_c5_rest = np.mean(proj_c5_rest, axis=0)
    
    # Compute discriminative score
    score_diff = np.abs((avg_c3_move - avg_c5_move) - (avg_c3_rest - avg_c5_rest))
    
    # Select top axes excluding specified components
    all_axes = np.arange(len(score_diff))
    excluded_axes = [0, 8, 9, 10]
    remaining_axes = np.setdiff1d(all_axes, excluded_axes)
    score_diff_remaining = score_diff[remaining_axes]
    top_relative = np.argsort(np.abs(score_diff_remaining))[::-1][:3]
    top_axes = remaining_axes[top_relative]
    
    print(score_diff)
    print(f"Top 3 PCA axes with highest discriminative power: {top_axes}")
    print("Their scores diff:", score_diff[top_axes])
    
    # Extract average waveforms from top axes
    average_waveforms = []
    for axis_idx in top_axes:
        component_scores = pca_transformed[:, axis_idx]
        bins = pd.cut(component_scores, bins=n_bins, labels=False, include_lowest=True)
        
        qs = [0, n_bins - 1]  # lowest and highest bins
        waveforms = []
        for q in qs:
            bin_indices = np.where(bins == q)[0]
            if len(bin_indices) == 0:
                avg_waveform = np.zeros_like(burst_array_waveform[0])
            else:
                avg_waveform = burst_array_waveform[bin_indices].mean(axis=0)
            waveforms.append(avg_waveform)
        average_waveforms.append(waveforms)
    
    average_waveforms = np.array(average_waveforms)
    average_waveforms = average_waveforms.reshape(-1, average_waveforms.shape[-1])
    print(f"Average waveforms shape: {average_waveforms.shape}")
    
    # Optional: Plot waveforms
    # (Plotting code commented out for production use)
    
    # Classification phase
    file_path = f"/mnt/data/Divyanshi/datasets_to_work/Perrine/Preprocessed/Notch_45/decim_4/{subject_type}_data"
    
    subject_scores = []
    std_scores = []
    subject_aucs = []
    auc_stds = []
    subjects = []
    per_run_aucs_score = []
    
    for subject in subject_range:
        subject_str = f"{subject:02d}"
        
        # Load movement epochs
        sub_base_path = os.path.join(file_path, "movement", f"sub_{subject_str}")
        index_path = os.path.join(sub_base_path, "indices", "random_trial_indices.npz")
        epochs_path = os.path.join(sub_base_path, f"epochs_{subject_type}_{subject_str}_movement_Notch_45.pkl")
        
        indices_file = np.load(index_path)
        indices = indices_file['trial_random']
        
        with open(epochs_path, "rb") as f:
            move_epochs = pickle.load(f)
        move_epochs = move_epochs.drop(indices)
        
        # Load rest epochs
        sub_base_path = os.path.join(file_path, "rest", f"sub_{subject_str}")
        index_path = os.path.join(sub_base_path, "indices", "random_trial_indices.npz")
        epochs_path = os.path.join(sub_base_path, f"epochs_{subject_type}_{subject_str}_rest_Notch_45.pkl")
        
        indices_file = np.load(index_path)
        indices = indices_file['trial_random']
        
        with open(epochs_path, "rb") as f:
            rest_epochs = pickle.load(f)
        rest_epochs = rest_epochs.drop(indices)
        
        # Get time axis and trim to analysis window
        time_axis = move_epochs.times
        start_idx = np.where(time_axis >= TIME_WINDOW_START)[0][0]
        end_idx = np.where(time_axis <= TIME_WINDOW_END)[0][-1] + 1
        
        X_movement = move_epochs.get_data()[:, :, start_idx:end_idx]
        X_rest = rest_epochs.get_data()[:, :, start_idx:end_idx]
        
        print(f"Trimmed time range: {time_axis[start_idx]:.2f}s to {time_axis[end_idx-1]:.2f}s")
        print(f"X_movement shape: {X_movement.shape}")
        print(f"X_rest shape: {X_rest.shape}")
        
        y_movement = np.ones(X_movement.shape[0])
        y_rest = np.zeros(X_rest.shape[0])
        
        X_all = np.concatenate([X_movement, X_rest], axis=0)
        y_all = np.concatenate([y_movement, y_rest], axis=0)
        
        print(f"X_all shape: {X_all.shape}")
        print(f"y_all shape: {y_all.shape}")
        
        if analysis_type == "burst_analysis":
            # Convolve epochs with burst waveform filters
            convolve_X_all = []
            for i in range(average_waveforms.shape[0]):
                convolved_trials = np.zeros_like(X_all)
                for j in range(X_all.shape[0]):
                    for k in range(X_all.shape[1]):
                        convolved_trials[j, k, :] = scipy.signal.convolve(
                            X_all[j, k, :],
                            average_waveforms[i],
                            mode='same',
                            method='direct'
                        )
                convolve_X_all.append(convolved_trials)
            
            for idx in range(len(convolve_X_all)):
                print(f"Shape of convolve_X_all[{idx}]: {convolve_X_all[idx].shape}")
            
            all_run_scores = []
            all_run_aucs = []
            
            for run in range(N_RUNS):
                print(f"\n==== Run {run + 1}/{N_RUNS} ====\n")
                
                skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=run)
                scores = []
                aucs = []
                
                for train_idx, test_idx in skf.split(convolve_X_all[0], y_all):
                    X_train_list = []
                    X_test_list = []
                    
                    for matrix in convolve_X_all:
                        X_train = matrix[train_idx]
                        X_test = matrix[test_idx]
                        y_train = y_all[train_idx]
                        y_test = y_all[test_idx]
                        
                        print("X_train", X_train.shape)
                        print("y_train", y_train.shape)
                        
                        csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, 
                                  norm_trace=False, component_order='mutual_info', 
                                  transform_into="average_power")
                        X_train_csp = csp.fit_transform(X_train, y_train)
                        X_test_csp = csp.transform(X_test)
                        
                        print(f"X_train_csp: {X_train_csp.shape}")
                        
                        X_train_list.append(X_train_csp)
                        X_test_list.append(X_test_csp)
                    
                    print("X_train", len(X_train_list))
                    print("X_test", len(X_test_list))
                    
                    X_train_concat = np.concatenate(X_train_list, axis=1)
                    print(f"X_train_concat: {X_train_concat.shape}")
                    X_test_concat = np.concatenate(X_test_list, axis=1)
                    print(f"X_test_concat: {X_test_concat.shape}")
                    
                    clf = LDA()
                    clf.fit(X_train_concat, y_train)
                    score = clf.score(X_test_concat, y_test)
                    scores.append(score)
                    print(f"Fold score: {score:.3f}")
                    
                    y_pred_proba = clf.predict_proba(X_test_concat)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    print("ROC AUC:", auc_score)
                    aucs.append(auc_score)
                
                all_run_scores.append(np.mean(scores))
                all_run_aucs.append(np.nanmean(aucs))
            
            mean_score = np.mean(all_run_scores)
            print(f"Mean CV score: {mean_score:.3f}")
            std_score = np.std(all_run_scores)
            mean_auc = np.mean(all_run_aucs)
            std_auc = np.std(all_run_aucs)
            
            subject_scores.append(mean_score)
            subject_aucs.append(mean_auc)
            std_scores.append(std_score)
            auc_stds.append(std_auc)
            per_run_aucs_score.append(all_run_aucs)
            subjects.append(subject)
            
            print(f"{subject_str}: Accuracy = {mean_score:.3f} ± {std_score:.3f}, AUC = {mean_auc:.3f} ± {std_auc:.3f}")
        
        elif analysis_type == "beta_analysis":
            all_run_scores = []
            all_run_aucs = []
            
            for run in range(N_RUNS):
                print(f"\n==== Run {run + 1}/{N_RUNS} ====\n")
                
                skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=run)
                scores = []
                aucs = []
                
                for train_idx, test_idx in skf.split(X_all, y_all):
                    X_train = X_all[train_idx]
                    X_test = X_all[test_idx]
                    y_train = y_all[train_idx]
                    y_test = y_all[test_idx]
                    
                    csp_features_all_bands_train = []
                    csp_features_all_bands_test = []
                    
                    for band_name, (fmin, fmax) in BETA_BANDS.items():
                        X_filtered_train = np.zeros_like(X_train)
                        for trial in range(X_train.shape[0]):
                            X_filtered_train[trial] = filter_data(X_train[trial], sfreq=SAMPLING_FREQ, 
                                                                   l_freq=fmin, h_freq=fmax, 
                                                                   fir_design='firwin', verbose=False)
                        
                        csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, 
                                  norm_trace=False, component_order='mutual_info', 
                                  transform_into="average_power")
                        X_train_csp = csp.fit_transform(X_filtered_train, y_train)
                        csp_features_all_bands_train.append(X_train_csp)
                        
                        X_filtered_test = np.zeros_like(X_test)
                        for trial in range(X_test.shape[0]):
                            X_filtered_test[trial] = filter_data(X_test[trial], sfreq=SAMPLING_FREQ, 
                                                                  l_freq=fmin, h_freq=fmax, 
                                                                  fir_design='firwin', verbose=False)
                        
                        X_test_csp = csp.transform(X_filtered_test)
                        csp_features_all_bands_test.append(X_test_csp)
                    
                    X_combined_train = np.concatenate(csp_features_all_bands_train, axis=1)
                    X_combined_test = np.concatenate(csp_features_all_bands_test, axis=1)
                    
                    clf = LDA()
                    clf.fit(X_combined_train, y_train)
                    score = clf.score(X_combined_test, y_test)
                    scores.append(score)
                    print(f"Fold score: {score:.3f}")
                    
                    y_pred_proba = clf.predict_proba(X_combined_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    print("ROC AUC:", auc_score)
                    aucs.append(auc_score)
                
                all_run_scores.append(np.mean(scores))
                all_run_aucs.append(np.nanmean(aucs))
            
            mean_score = np.mean(all_run_scores)
            print(f"Mean CV score: {mean_score:.3f}")
            std_score = np.std(all_run_scores)
            mean_auc = np.mean(all_run_aucs)
            std_auc = np.std(all_run_aucs)
            
            subject_scores.append(mean_score)
            subject_aucs.append(mean_auc)
            std_scores.append(std_score)
            auc_stds.append(std_auc)
            per_run_aucs_score.append(all_run_aucs)
            subjects.append(subject)
            
            print(analysis_type, f"{subject_str}: Accuracy = {mean_score:.3f} ± {std_score:.3f}, AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    
    return (subject_scores, subjects, std_scores, subject_aucs, auc_stds, per_run_aucs_score)


if __name__ == "__main__":
    # Run both analysis types
    subject_scores, subjects, std_scores, subject_aucs, auc_stds, per_run_aucs_beta = run_analysis("Patient", "beta_analysis")
    subject_scores_2, subjects, std_scores_2, subject_aucs_2, auc_stds_2, per_run_aucs_burst = run_analysis("Patient", "burst_analysis")
    
    # Previous save format (kept for reference - includes all metrics)
    '''np.savez("analysis_07_07_25_p.npz", 
        subject_scores=subject_scores, 
        subjects=subjects, 
        std_scores=std_scores, 
        subject_aucs=subject_aucs, 
        auc_stds=auc_stds, 
        subject_scores_2=subject_scores_2, 
        subjects_2=subjects, 
        std_scores_2=std_scores_2, 
        subject_aucs_2=subject_aucs_2, 
        auc_stds_2=auc_stds_2)'''
    
    # Current save format - focused on AUC metrics and per-run data
    np.savez("analysis_08_07_25_P.npz", 
        subject_aucs=np.array(subject_aucs), 
        subject_aucs_2=np.array(subject_aucs_2), 
        std_scores=np.array(std_scores), 
        auc_stds=np.array(auc_stds), 
        per_run_aucs_beta=np.array(per_run_aucs_beta),
        per_run_aucs_burst=np.array(per_run_aucs_burst))
    
    print("Analysis results for subjects with 6 bands saved successfully.")
