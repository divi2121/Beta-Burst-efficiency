"""
EEG preprocessing pipeline for motor imagery/intention tasks.

Handles loading, filtering, epoching, artifact rejection, and saving
of preprocessed EEG data for both movement and rest conditions.
"""

import os
from os.path import join, exists
import sys
import pickle

import numpy as np
import mne
from autoreject import AutoReject, get_rejection_threshold

# Local imports (these files must be present in the repository)
# from zapline_iter import zapline_until_gone
# from help_funcs import load_exp_variables
# from burst_analysis import TfBursts


def get_preprocessed_epochs(subject_id, condition):
    """
    Generate preprocessed Epochs objects for the specified condition and subject.
    
    Returns:
        labels_dict: dict with keys 'movement', 'rest' containing label arrays
        processed_epochs_dict: dict with keys 'movement', 'rest' containing mne.Epochs
        condition: str, the preprocessing condition used
    """
    # Load raw data
    fif_file = f"/mnt/data/Divyanshi/datasets_to_work/Perrine/Raw/_data/FIF/P{subject_id}/P{subject_id}_Claassen.raw.fif"
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    raw.filter(l_freq=0, h_freq=120)
    sfreq = raw.info['sfreq']
    
    # Apply condition-specific filtering
    if condition == "Notch_45":
        raw.filter(l_freq=0, h_freq=120)
        raw.notch_filter(50)
    elif condition == "45_BP":
        raw.filter(l_freq=0, h_freq=120)
    
    # Define event IDs
    event_id_rest = {'Stimulus/S 11': 11, 'Stimulus/S 12': 12}
    event_id_movement = {'Stimulus/S 1': 1, 'Stimulus/S 2': 2}
    
    # Extract events
    events, _ = mne.events_from_annotations(raw)
    
    # Epoching parameters
    tmin, tmax = 0, 12.0
    tmin_rest, tmax_rest = 0, 12.0
    
    # Create epochs
    epochs_dict = {'movement': [], 'rest': []}
    
    movement_epochs = mne.Epochs(raw, events, event_id_movement, tmin=tmin, tmax=tmax,
                                  baseline=None, preload=True, decim=4)
    epochs_dict['movement'] = movement_epochs
    
    rest_epochs = mne.Epochs(raw, events, event_id_rest, tmin=tmin_rest, tmax=tmax_rest,
                              baseline=None, preload=True, decim=4)
    epochs_dict['rest'] = rest_epochs
    
    # Process each condition
    labels_dict = {}
    processed_epochs_dict = {}
    
    for lbl, epochs in epochs_dict.items():
        # Channel selection
        epochs.pick(['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'CP5', 'CP6', 'Pz'])
        
        # Bandpass filter
        epochs.filter(l_freq=1, h_freq=45)
        
        # AutoReject for artifact rejection
        ar = AutoReject(n_jobs=-1)
        reject = get_rejection_threshold(epochs, decim=2)
        epochs.reject = None
        print(f"Subject {subject_id}: {reject}")
        epochs.drop_bad(reject=reject)
        
        # Print rejection summary
        drop_log = epochs.drop_log
        rejected_trials = [i for i, log in enumerate(drop_log) if log != ()]
        rejection_reasons = [log for log in drop_log if log != ()]
        
        if rejected_trials:
            print(f"Subject {subject_id}")
            for trial, reason in zip(rejected_trials, rejection_reasons):
                print(f" - Trial {trial} rejected due to: {', '.join(reason)}")
        else:
            print(f"Subject {subject_id} No trials rejected.")
        
        # Extract labels
        if lbl == 'movement':
            hand_labels = []
            for event in epochs.events:
                event_id = event[2]
                if event_id == 1:
                    hand_labels.append('left')
                elif event_id == 2:
                    hand_labels.append('right')
            labels_dict['movement'] = np.array(hand_labels)
        elif lbl == 'rest':
            labels_dict['rest'] = np.array(['rest'] * len(epochs))
        
        processed_epochs_dict[lbl] = epochs
    
    return labels_dict, processed_epochs_dict, condition


def save_data(file_path, data):
    """Save data as a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def process_single_subject(subject_id, condition, data_type):
    """
    Process and save movement and rest epochs for a given subject.
    
    Args:
        subject_id: int, subject ID
        condition: str, preprocessing condition (e.g., 'Notch_45', '45_BP')
        data_type: str, 'Patient' or 'Subject'
    
    Returns:
        Tuple of (subject_id_str, movement_epochs, rest_epochs, movement_labels, rest_labels, data_dir)
    """
    subject_id_str = f"{subject_id:02d}"
    
    # Get preprocessed epochs
    labels_dict, processed_epochs_dict, condition = get_preprocessed_epochs(
        subject_id_str, condition=condition)
    
    # Setup directory structure
    base_dir = f"/mnt/data/Divyanshi/datasets_to_work/Perrine/Preprocessed/{condition}/decim_4"
    
    if data_type == "Patient":
        data_dir = os.path.join(base_dir, "Patient_data")
    elif data_type == "Subject":
        data_dir = os.path.join(base_dir, "Subject_data")
    
    movement_dir = os.path.join(data_dir, "movement", f"sub_{subject_id_str}")
    rest_dir = os.path.join(data_dir, "rest", f"sub_{subject_id_str}")
    
    # Create directories
    os.makedirs(movement_dir, exist_ok=True)
    os.makedirs(rest_dir, exist_ok=True)
    
    # Save movement data
    movement_epochs = processed_epochs_dict['movement']
    movement_labels = labels_dict['movement']
    
    epoch_file = os.path.join(movement_dir, f"epochs_{data_type}_{subject_id_str}_movement_{condition}.pkl")
    label_file = os.path.join(movement_dir, f"labels_{data_type}_{subject_id_str}_movement_{condition}.pkl")
    
    save_data(epoch_file, movement_epochs)
    save_data(label_file, movement_labels)
    
    print(f"Saved movement epochs for {data_type} {subject_id_str}")
    print(f"Saved movement labels for {data_type} {subject_id_str}")
    
    # Save rest data
    rest_epochs = processed_epochs_dict['rest']
    rest_labels = labels_dict['rest']
    
    epoch_file = os.path.join(rest_dir, f"epochs_{data_type}_{subject_id_str}_rest_{condition}.pkl")
    label_file = os.path.join(rest_dir, f"labels_{data_type}_{subject_id_str}_rest_{condition}.pkl")
    
    save_data(epoch_file, rest_epochs)
    save_data(label_file, rest_labels)
    
    print(f"Saved rest epochs for {data_type} {subject_id_str}")
    print(f"Saved rest labels for {data_type} {subject_id_str}")
    
    return subject_id_str, movement_epochs, rest_epochs, movement_labels, rest_labels, data_dir


def setup_burst_analysis(experimental_vars, freq_step=0.5, tf_method="superlets",
                          produce_plots=False, plot_format="png", remove_fooof=False):
    """
    Set up TfBursts object for burst extraction.
    
    Note: Requires TfBursts class from burst_analysis module.
    """
    freqs = np.arange(1.0, 43.25, freq_step)
    upto_gamma_band = np.array([1, 40])
    upto_gamma_range = np.where(
        np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
    )[0]
    
    # This requires burst_analysis.TfBursts to be available
    # Commented out since it's a dependency
    '''
    bm = TfBursts(
        experimental_vars,
        freqs,
        upto_gamma_band,
        upto_gamma_range,
        tf_method=tf_method,
        produce_plots=produce_plots,
        plot_format=plot_format,
        remove_fooof=remove_fooof,
    )
    return bm
    '''
    pass


def extract_correct_burst(bm, subject_id_str, movement_epochs, rest_epochs,
                           movement_labels, rest_labels, move_or_rest):
    """
    Extract bursts based on movement or rest condition.
    
    Note: Requires TfBursts object from burst_analysis module.
    """
    if move_or_rest == "movement":
        # bm.burst_extraction(subject_id_str, movement_epochs.get_data(), movement_labels, band="beta")
        print("Movement burst extracted.")
    elif move_or_rest == "rest":
        # bm.burst_extraction(subject_id_str, rest_epochs.get_data(), rest_labels, band="beta")
        print("Rest burst extracted.")


if __name__ == "__main__":
    # Get subject ID from command line
    sid = int(sys.argv[1])
    
    # Process subject
    subject_id_str, movement_epochs, rest_epochs, movement_labels, rest_labels, data_dir = process_single_subject(
        sid, 'Notch_45', 'Patient')
    
    print(f"Processed Subject {subject_id_str}")
    
    # Burst extraction would go here (requires additional dependencies)
    # move_or_rest = "rest"
    # variables_path = os.path.join(data_dir, move_or_rest, "variables-2.json")
    # experimental_vars = load_exp_variables(json_filename=variables_path)
    # bm = setup_burst_analysis(experimental_vars, ...)
    # extract_correct_burst(bm, subject_id_str, movement_epochs, rest_epochs, 
    #                       movement_labels, rest_labels, move_or_rest)
    
    print("done")
