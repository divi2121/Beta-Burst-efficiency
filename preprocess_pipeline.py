#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_pipeline.py

Modular preprocessing pipeline for EEG motor-imagery / motor-attempt data.

This script implements:
- loading an MNE FIF raw file
- initial wide-band filtering
- event extraction
- epoching for movement and rest conditions
- (conditionally) zapline / notch removal for line noise
- bandpass filtering (1-45 Hz)
- channel selection
- compute rejection thresholds (autoreject optional)
- label extraction (left/right/rest)
- save preprocessed epochs & labels as pickles

Notes:
- This script **does not** include raw clinical data. Paths are configurable.
- Helper modules required from the repo:
    - zapline_iter.py  (function: zapline_until_gone)
    - help_funcs.py    (function: load_exp_variables)  [only used if you want variables JSON]
- AutoReject usage is optional and by default only computes thresholds without applying repair.
"""

import os
import argparse
import logging
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import mne

# optional heavy dependency; import may fail if not installed
try:
    from autoreject import AutoReject, get_rejection_threshold
    AUTOREJECT_AVAILABLE = True
except Exception:
    AUTOREJECT_AVAILABLE = False

# local helper modules (must be present in repo)
from zapline_iter import zapline_until_gone  # expects (data, target_freq, sfreq, win_sz, spot_sz) -> (cleaned, meta)
# from help_funcs import load_exp_variables  # only if you want to load variable JSONs

# -----------------------
# Configuration / paths
# -----------------------
BASE_RAW_PATH = "/mnt/data/Divyanshi/datasets_to_work/Perrine/Raw/_data/FIF"
BASE_PREPROCESSED = "/mnt/data/Divyanshi/datasets_to_work/Perrine/Preprocessed"

REQUESTED_CHANNELS = ["F3", "Fz", "F4", "C3", "Cz", "C4", "CP5", "CP6", "Pz"]
DEFAULT_SFREQ_EXPECTED = 1000  # typical; zapline uses actual sfreq from file

# Logging
logging.basicConfig(level=logging.INFO, fmt="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("preprocess_pipeline")


# -----------------------
# Utility functions
# -----------------------
def save_pickle(path: str, obj: Any) -> None:
    """Save an object to a pickle file, creating parent dir if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved pickle: {path}")


# -----------------------
# Core pipeline steps
# -----------------------
def load_raw_fif(fif_path: str, preload: bool = True) -> mne.io.Raw:
    """Load an MNE raw FIF file."""
    if not os.path.exists(fif_path):
        raise FileNotFoundError(f"FIF file not found: {fif_path}")
    logger.info(f"Loading raw FIF: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)
    return raw


def initial_filter(raw: mne.io.Raw, l_freq: float = 0.0, h_freq: float = 120.0) -> None:
    """Apply a broad initial filter on the raw object (in-place)."""
    logger.info(f"Applying initial filter: {l_freq} - {h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)


def extract_events(raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return events array and a default event_id mapping (documented mapping)."""
    logger.info("Extracting events/annotations from raw data")
    events, event_id = mne.events_from_annotations(raw)
    # Document expected mapping (this is descriptive; change if yours differs)
    expected_map = {
        "Stimulus/S  1": 1,   # movement left
        "Stimulus/S  2": 2,   # movement right
        "Stimulus/S  11": 11, # rest
        "Stimulus/S  12": 12, # rest
    }
    return events, expected_map


def make_epochs(raw: mne.io.Raw, events: np.ndarray, event_id_map: Dict[str, int], decim: int = 4) -> Dict[str, mne.Epochs]:
    """
    Create movement and rest Epochs objects with recommended time windows.
    Returns dict: {'movement': Epochs, 'rest': Epochs}
    """
    logger.info("Creating movement and rest epochs")
    # Movement: longer window to capture attempts (-2 to +14s)
    movement_event_ids = {k: v for k, v in event_id_map.items() if v in (1, 2)}
    rest_event_ids = {k: v for k, v in event_id_map.items() if v in (11, 12)}

    tmin_mov, tmax_mov = -2.0, 14.0
    tmin_rest, tmax_rest = 0.0, 8.0

    movement_epochs = mne.Epochs(raw, events, movement_event_ids, tmin=tmin_mov, tmax=tmax_mov,
                                 baseline=None, preload=True, decim=decim, verbose=False)
    rest_epochs = mne.Epochs(raw, events, rest_event_ids, tmin=tmin_rest, tmax=tmax_rest,
                             baseline=None, preload=True, decim=decim, verbose=False)

    return {"movement": movement_epochs, "rest": rest_epochs}


def select_channels(epochs: mne.Epochs, requested: list = REQUESTED_CHANNELS) -> None:
    """Pick only channels from requested list that exist in epoch; raises error if none found."""
    picks = [ch for ch in requested if ch in epochs.ch_names]
    if not picks:
        raise RuntimeError("None of the requested channels present in the data: " + ", ".join(requested))
    logger.info(f"Selecting channels: {picks}")
    epochs.pick(picks)


def apply_zapline_condition(epochs: mne.Epochs, sfreq: float, target_freq: int = 50, noise_wins: tuple = (10, 5)):
    """
    Apply zapline_until_gone on epoch data (in-place replacement of data).
    zapline_until_gone should accept array with shape (n_epochs, n_channels, n_times) or similar.
    """
    logger.info("Applying zapline (line-noise removal)")
    data = epochs.get_data()  # shape: n_epochs x n_channels x n_times
    cleaned, meta = zapline_until_gone(data, target_freq=target_freq, sfreq=int(round(sfreq)), win_sz=noise_wins[0], spot_sz=noise_wins[1])
    # Ensure contiguous and correct shape
    epochs._data = np.ascontiguousarray(cleaned)
    logger.debug("Zapline applied and epoch data overwritten")


def bandpass_epochs(epochs: mne.Epochs, l_freq: float = 1.0, h_freq: float = 45.0) -> None:
    """Bandpass filter epochs in-place for analysis band (1-45 Hz by default)."""
    logger.info(f"Bandpass filtering epochs: {l_freq} - {h_freq} Hz")
    epochs.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)


def compute_rejection_thresholds(epochs: mne.Epochs, decim: int = 2) -> dict:
    """Compute rejection thresholds using autoreject utility (if available)."""
    logger.info("Computing rejection thresholds (autoreject helper)")
    if not AUTOREJECT_AVAILABLE:
        logger.warning("autoreject not installed: skipping threshold computation")
        return {}
    try:
        thresholds = get_rejection_threshold(epochs, decim=decim)
        logger.debug(f"Rejection thresholds: {thresholds}")
        return thresholds
    except Exception as e:
        logger.warning(f"Failed to compute rejection thresholds: {e}")
        return {}


def optional_autoreject_fit(epochs: mne.Epochs, n_jobs: int = 1) -> None:
    """Optionally fit AutoReject and repair epochs (use with caution)."""
    if not AUTOREJECT_AVAILABLE:
        raise RuntimeError("AutoReject not available. Install `autoreject` to use this function.")
    logger.info("Fitting AutoReject (this may take time)")
    ar = AutoReject(n_jobs=n_jobs)
    ar.fit(epochs)
    # apply repair / reject depends on usage; here we just log that we fitted
    logger.info("AutoReject fitted (not applied automatically here).")


def extract_labels_from_epochs(epochs: mne.Epochs, label_type: str) -> np.ndarray:
    """Extract labels array for a given epoch object. label_type: 'movement' or 'rest'"""
    if label_type == "movement":
        hand_labels = []
        for ev in epochs.events:
            eid = int(ev[2])
            if eid == 1:
                hand_labels.append("left")
            elif eid == 2:
                hand_labels.append("right")
            else:
                hand_labels.append("unknown")
        labels = np.array(hand_labels)
    else:
        labels = np.array(["rest"] * len(epochs))
    logger.info(f"Extracted {len(labels)} labels for {label_type}")
    return labels


# -----------------------
# High level processing
# -----------------------
def process_subject(subject_id: int, condition: str = "ZAP_45_BP", data_type: str = "Patient") -> Tuple[str, Dict[str, mne.Epochs], Dict[str, np.ndarray], str]:
    """
    End-to-end preprocessing for a single subject.
    Returns:
      subject_id_str, epochs_dict, labels_dict, data_dir
    """
    subject_id_str = f"{int(subject_id):02d}"
    logger.info(f"Starting preprocessing for subject {subject_id_str} | condition={condition} | data_type={data_type}")

    fif_path = os.path.join(BASE_RAW_PATH, f"P{subject_id_str}", f"P{subject_id_str}_Claassen.raw.fif")
    raw = load_raw_fif(fif_path)
    initial_filter(raw, l_freq=0.0, h_freq=120.0)

    events, event_map = extract_events(raw)
    epochs_dict = make_epochs(raw, events, event_map, decim=4)

    labels_dict = {}
    processed_epochs = {}

    for lbl, epochs in epochs_dict.items():
        logger.info(f"Preprocessing epoch set: {lbl}")

        # channel selection (safe)
        select_channels(epochs, REQUESTED_CHANNELS)

        # conditional line-noise removal
        if condition == "ZAP_45_BP":
            sfreq = raw.info.get("sfreq", DEFAULT_SFREQ_EXPECTED)
            # zapline modifies epoch data in-place
            apply_zapline_condition(epochs, sfreq=sfreq, target_freq=50, noise_wins=(10, 5))

        # bandpass for analysis
        bandpass_epochs(epochs, l_freq=1.0, h_freq=45.0)

        # compute rejection thresholds (autoreject helper)
        thresholds = compute_rejection_thresholds(epochs, decim=2)

        # (Optional) fit AutoReject if desired and available
        # optional_autoreject_fit(epochs, n_jobs=1)

        # extract labels
        labels = extract_labels_from_epochs(epochs, lbl)

        processed_epochs[lbl] = epochs
        labels_dict[lbl] = labels

    # Prepare saving directories
    base_dir = os.path.join(BASE_PREPROCESSED, condition, "decim_4")
    data_dir = os.path.join(base_dir, "patient_data" if data_type == "Patient" else "subject_data")
    movement_dir = os.path.join(data_dir, "movement", f"sub_{subject_id_str}")
    rest_dir = os.path.join(data_dir, "rest", f"sub_{subject_id_str}")
    os.makedirs(movement_dir, exist_ok=True)
    os.makedirs(rest_dir, exist_ok=True)

    # Save results
    mov_epoch_file = os.path.join(movement_dir, f"epochs_{data_type}_{subject_id_str}_movement_{condition}.pkl")
    mov_label_file = os.path.join(movement_dir, f"labels_{data_type}_{subject_id_str}_movement_{condition}.pkl")
    save_pickle(mov_epoch_file, processed_epochs["movement"])
    save_pickle(mov_label_file, labels_dict["movement"])

    rest_epoch_file = os.path.join(rest_dir, f"epochs_{data_type}_{subject_id_str}_rest_{condition}.pkl")
    rest_label_file = os.path.join(rest_dir, f"labels_{data_type}_{subject_id_str}_rest_{condition}.pkl")
    save_pickle(rest_epoch_file, processed_epochs["rest"])
    save_pickle(rest_label_file, labels_dict["rest"])

    logger.info(f"Saved preprocessed outputs for subject {subject_id_str} in {data_dir}")
    return subject_id_str, processed_epochs, labels_dict, data_dir


# -----------------------
# CLI
# -----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Preprocessing pipeline for EEG subjects (movement/rest).")
    p.add_argument("--subject", "-s", required=True, help="Subject id (int).")
    p.add_argument("--condition", "-c", default="ZAP_45_BP", choices=["ZAP_45_BP", "45_BP"], help="Preprocessing condition.")
    p.add_argument("--data-type", "-d", default="Patient", choices=["Patient", "Subject"], help="Data type.")
    p.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level))
    sid = int(args.subject)
    condition = args.condition
    data_type = args.data_type

    try:
        subject_id_str, epochs, labels, data_dir = process_subject(sid, condition=condition, data_type=data_type)
        logger.info(f"Preprocessing complete for subject {subject_id_str}")
    except Exception as e:
        logger.exception(f"Processing failed for subject {args.subject}: {e}")
        raise


if __name__ == "__main__":
    main()
