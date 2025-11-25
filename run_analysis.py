#!/usr/bin/env python3
"""
Refactored run_analysis with minimal, high-impact fixes for correctness and robustness.

What's new (minimum to commit):
- validate_config(cfg): checks required config fields and types, raises informative errors.
- sizes_per_subject collected while stacking per-subject waveforms.
- build_global_class_indices(sizes_per_subject): constructs PCA row indices per class
  in the same order as the stacked burst_array, avoiding PCA/class-index misalignment.
- Save sizes_per_subject and top_axes into the results .npz for traceability.

Other small clarity improvements preserved from the Priority 1 refactor:
- meaningful variable names, pathlib usage, logging, docstrings on helpers.

Intended next steps (not implemented here): add unit tests (a simple test file is provided separately),
add CLI arg parsing, and integrate synthetic-data generation into CI.
"""
from pathlib import Path
import json
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from mne.decoding import CSP
from mne.filter import filter_data
import scipy.signal

# Configure module logger
logger = logging.getLogger("run_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------- Config validation ---------------------- #
def validate_config(cfg: dict) -> dict:
    """
    Minimal validation for config dict. Raises ValueError with a clear message if invalid.
    Returns cfg (unchanged) if validation passes.

    Required structure (minimal):
      cfg["paths"]["preprocessed_dir"] (str)
      cfg["paths"]["condition"] (str)
      cfg["paths"]["decim_dir"] (str)
      cfg["eeg"]["sfreq"] (number)
      cfg["eeg"]["time_window"] (length-2 iterable)
      cfg["analysis"]["pca_components"] (int)
      cfg["analysis"]["pca_bins"] (int)
      cfg["analysis"]["n_splits"] (int)
      cfg["analysis"]["n_runs"] (int)
      cfg["analysis"]["bands"] (dict-like)
      cfg["subjects"]["patient_range"] (length-2 iterable)
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    # paths
    paths = cfg.get("paths")
    if not paths or not all(k in paths for k in ("preprocessed_dir", "condition", "decim_dir")):
        raise ValueError("config['paths'] must include 'preprocessed_dir', 'condition', 'decim_dir'")

    # eeg
    eeg = cfg.get("eeg")
    if not eeg or "sfreq" not in eeg or "time_window" not in eeg:
        raise ValueError("config['eeg'] must include 'sfreq' and 'time_window'")
    if not isinstance(eeg["sfreq"], (int, float)):
        raise ValueError("config['eeg']['sfreq'] must be a number")
    if not hasattr(eeg["time_window"], "__iter__") or len(eeg["time_window"]) != 2:
        raise ValueError("config['eeg']['time_window'] must be an iterable of length 2")

    # analysis
    analysis = cfg.get("analysis")
    if not analysis:
        raise ValueError("config must include 'analysis' section")
    required_analysis = ("pca_components", "pca_bins", "n_splits", "n_runs", "bands")
    for k in required_analysis:
        if k not in analysis:
            raise ValueError(f"config['analysis'] missing required key: {k}")
    if not isinstance(analysis["pca_components"], int) or analysis["pca_components"] < 1:
        raise ValueError("config['analysis']['pca_components'] must be a positive int")
    if not isinstance(analysis["pca_bins"], int) or analysis["pca_bins"] < 2:
        raise ValueError("config['analysis']['pca_bins'] must be an int >= 2")
    if not isinstance(analysis["n_splits"], int) or analysis["n_splits"] < 2:
        raise ValueError("config['analysis']['n_splits'] must be an int >= 2")
    if not isinstance(analysis["n_runs"], int) or analysis["n_runs"] < 1:
        raise ValueError("config['analysis']['n_runs'] must be an int >= 1")
    if not isinstance(analysis["bands"], dict) or len(analysis["bands"]) == 0:
        raise ValueError("config['analysis']['bands'] must be a non-empty dict")

    # subjects
    subjects = cfg.get("subjects")
    if not subjects or "patient_range" not in subjects:
        raise ValueError("config['subjects'] must include 'patient_range'")
    pr = subjects["patient_range"]
    if not hasattr(pr, "__iter__") or len(pr) != 2:
        raise ValueError("config['subjects']['patient_range'] must be an iterable of length 2")

    return cfg


# ---------------------- Small helpers ---------------------- #
def load_config(path: str = "config.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        cfg = json.load(f)
    return validate_config(cfg)


def _safe_load_npy(path: Path):
    arr = np.load(path, allow_pickle=True)
    try:
        if isinstance(arr, np.ndarray) and arr.shape == ():
            return arr.item()
    except Exception:
        pass
    return arr


def get_subjects_for_type(subject_type: str, cfg: dict):
    if subject_type == "Patient":
        pr = cfg["subjects"]["patient_range"]
        excludes = cfg["subjects"].get("patient_exclude", [])
        return [i for i in range(pr[0], pr[1]) if i not in excludes]
    elif subject_type == "Subject":
        sr = cfg["subjects"]["subject_range"]
        return list(range(sr[0], sr[1] + 1))
    else:
        raise ValueError("subject_type must be 'Patient' or 'Subject'")


def select_random_trials_and_save(data: dict, subsample: float, save_dir: Path, rng: np.random.Generator) -> np.ndarray:
    trials_c3 = np.unique(data[3]["trial"])
    trials_c5 = np.unique(data[5]["trial"])
    common = np.intersect1d(trials_c3, trials_c5)
    if common.size == 0:
        selected = np.array([], dtype=int)
    else:
        n_sample = max(1, int(len(common) * subsample))
        selected = rng.choice(common, size=n_sample, replace=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / "random_trial_indices.npz", trial_random=selected)
    return selected


def extract_waveforms(data: dict, trial_ids: np.ndarray, tmin: float, tmax: float):
    if trial_ids.size == 0:
        return (np.empty((0, 0)), np.empty((0, 0)), np.empty(0), np.empty(0))

    def gather_indices(channel_idx):
        idxs = []
        for t in trial_ids:
            found = np.where(data[channel_idx]["trial"] == t)[0]
            if found.size > 0:
                idxs.append(found)
        if not idxs:
            return np.array([], dtype=int)
        return np.hstack(idxs)

    idx3 = gather_indices(3)
    idx5 = gather_indices(5)

    wf3 = data[3]["waveform"][idx3] if idx3.size > 0 else np.empty((0, data[3]["waveform"].shape[1]))
    pt3 = data[3]["peak_time"][idx3] if idx3.size > 0 else np.empty((0,))
    wf5 = data[5]["waveform"][idx5] if idx5.size > 0 else np.empty((0, data[5]["waveform"].shape[1]))
    pt5 = data[5]["peak_time"][idx5] if idx5.size > 0 else np.empty((0,))

    mask3 = (pt3 >= tmin) & (pt3 <= tmax)
    mask5 = (pt5 >= tmin) & (pt5 <= tmax)

    return wf3[mask3], wf5[mask5], pt3[mask3], pt5[mask5]


def build_global_class_indices(sizes_per_subject: List[Tuple[int, int, int, int]]):
    """
    From sizes_per_subject (list of 4-tuples per subject: (n_c3_move,n_c5_move,n_c3_rest,n_c5_rest)),
    build global lists of row indices (into the stacked burst_array) for each class in the same order
    used to stack per-subject arrays: [c3_move, c5_move, c3_rest, c5_rest] for each subject.
    """
    c3_move_idx, c5_move_idx, c3_rest_idx, c5_rest_idx = [], [], [], []
    offset = 0
    for (n0, n1, n2, n3) in sizes_per_subject:
        if n0 > 0:
            c3_move_idx.extend(range(offset, offset + n0))
        offset += n0
        if n1 > 0:
            c5_move_idx.extend(range(offset, offset + n1))
        offset += n1
        if n2 > 0:
            c3_rest_idx.extend(range(offset, offset + n2))
        offset += n2
        if n3 > 0:
            c5_rest_idx.extend(range(offset, offset + n3))
        offset += n3
    return (
        np.array(c3_move_idx, dtype=int),
        np.array(c5_move_idx, dtype=int),
        np.array(c3_rest_idx, dtype=int),
        np.array(c5_rest_idx, dtype=int),
    )


# ---------------------- Main analysis ---------------------- #
def run_analysis(subject_type: str, analysis_type: str, config_path: str = "config.json", random_seed: int = 0):
    cfg = load_config(config_path)
    rng = np.random.default_rng(random_seed)

    base_preprocessed_dir = Path(cfg["paths"]["preprocessed_dir"])
    condition = cfg["paths"]["condition"]
    decim = cfg["paths"]["decim_dir"]

    sfreq = cfg["eeg"]["sfreq"]
    tmin, tmax = cfg["eeg"]["time_window"]

    pca_components = cfg["analysis"]["pca_components"]
    pca_bins = cfg["analysis"]["pca_bins"]
    excluded_axes = cfg["analysis"].get("excluded_axes", [])
    n_splits = cfg["analysis"]["n_splits"]
    n_runs = cfg["analysis"]["n_runs"]
    bands = cfg["analysis"]["bands"]

    subsample = cfg["analysis"].get("subsample", 0.2)

    subjects = get_subjects_for_type(subject_type, cfg)

    # accumulators
    all_sub_waveforms = []
    burst_list_waveform_c3_move = []
    burst_list_waveform_c5_move = []
    burst_list_waveform_c3_rest = []
    burst_list_waveform_c5_rest = []
    peak_times_move = []
    peak_times_rest = []
    sizes_per_subject: List[Tuple[int, int, int, int]] = []

    processed_subjects = []

    # Loop subjects
    for subject in subjects:
        subject_str = f"{subject:02d}"
        sub_dir_move = base_preprocessed_dir / condition / decim / f"{subject_type}_data" / "movement" / f"sub_{subject_str}"
        sub_dir_rest = base_preprocessed_dir / condition / decim / f"{subject_type}_data" / "rest" / f"sub_{subject_str}"

        fn_move = sub_dir_move / "beta_bursts_superlets_nfs.npy"
        fn_rest = sub_dir_rest / "beta_bursts_superlets_nfs.npy"
        if not (fn_move.exists() and fn_rest.exists()):
            logger.warning("Missing burst files for subject %s, skipping", subject_str)
            continue

        data_move = _safe_load_npy(fn_move)
        data_rest = _safe_load_npy(fn_rest)

        # Select random trials and save indices in subject directories
        move_indices_dir = sub_dir_move / "indices"
        rest_indices_dir = sub_dir_rest / "indices"
        trial_random_move = select_random_trials_and_save(data_move, subsample, move_indices_dir, rng)
        trial_random_rest = select_random_trials_and_save(data_rest, subsample, rest_indices_dir, rng)

        wf_c3_move, wf_c5_move, pt_c3_move, pt_c5_move = extract_waveforms(data_move, trial_random_move, tmin, tmax)
        wf_c3_rest, wf_c5_rest, pt_c3_rest, pt_c5_rest = extract_waveforms(data_rest, trial_random_rest, tmin, tmax)

        # skip subject if nothing found
        if wf_c3_move.size == 0 and wf_c5_move.size == 0 and wf_c3_rest.size == 0 and wf_c5_rest.size == 0:
            logger.info("No valid bursts for subject %s, skipping", subject_str)
            continue

        # scale per subject and append
        scaler = RobustScaler()
        waveforms_all = np.vstack((wf_c3_move, wf_c5_move, wf_c3_rest, wf_c5_rest))
        waveforms_trans = scaler.fit_transform(waveforms_all)
        all_sub_waveforms.append(waveforms_trans)

        # compute slices to append to per-class lists (preserve order)
        n_c3_move = wf_c3_move.shape[0]
        n_c5_move = wf_c5_move.shape[0]
        n_c3_rest = wf_c3_rest.shape[0]
        n_c5_rest = wf_c5_rest.shape[0]

        burst_list_waveform_c3_move.append(waveforms_trans[:n_c3_move, :])
        burst_list_waveform_c5_move.append(waveforms_trans[n_c3_move:n_c3_move + n_c5_move, :])
        start_rest = n_c3_move + n_c5_move
        burst_list_waveform_c3_rest.append(waveforms_trans[start_rest:start_rest + n_c3_rest, :])
        burst_list_waveform_c5_rest.append(waveforms_trans[start_rest + n_c3_rest:, :])

        peak_times_move += [pt_c3_move, pt_c5_move]
        peak_times_rest += [pt_c3_rest, pt_c5_rest]

        sizes_per_subject.append((n_c3_move, n_c5_move, n_c3_rest, n_c5_rest))
        processed_subjects.append(subject)
        logger.info("Processed subject %s: c3_move=%d c5_move=%d c3_rest=%d c5_rest=%d",
                    subject_str, n_c3_move, n_c5_move, n_c3_rest, n_c5_rest)

    if len(all_sub_waveforms) == 0:
        logger.error("No burst waveforms collected for any subject")
        return [], [], [], [], []

    # PCA on stacked waveforms
    burst_array = np.vstack(all_sub_waveforms)
    pca = PCA(n_components=pca_components)
    pca_trans = pca.fit_transform(burst_array)

    # clip extreme percentiles per PCA axis
    for i in range(pca_components):
        lower, upper = np.percentile(pca_trans[:, i], [2, 98])
        pca_trans[:, i] = np.clip(pca_trans[:, i], lower, upper)

    # Build class indices robustly from sizes_per_subject
    idx_c3_move, idx_c5_move, idx_c3_rest, idx_c5_rest = build_global_class_indices(sizes_per_subject)

    # compute class means using indices that align to burst_array stacking
    def safe_mean(indices):
        return np.mean(pca_trans[indices, :], axis=0) if indices.size > 0 else np.zeros((pca_components,))

    mean_c3_move = safe_mean(idx_c3_move)
    mean_c5_move = safe_mean(idx_c5_move)
    mean_c3_rest = safe_mean(idx_c3_rest)
    mean_c5_rest = safe_mean(idx_c5_rest)

    score_diff = np.abs((mean_c3_move - mean_c5_move) - (mean_c3_rest - mean_c5_rest))

    remaining_axes = np.setdiff1d(np.arange(len(score_diff)), cfg["analysis"].get("excluded_axes", []))
    top_axes = remaining_axes[np.argsort(np.abs(score_diff[remaining_axes]))[::-1][:3]]

    logger.info("Top PCA axes: %s, scores: %s", top_axes.tolist(), score_diff[top_axes].tolist())

    # Classification (only beta_analysis implemented here)
    subject_scores, std_scores, subject_aucs, auc_stds = [], [], [], []

    for subject in processed_subjects:
        subject_str = f"{subject:02d}"
        base_path = base_preprocessed_dir / condition / decim / f"{subject_type}_data"
        sub_dir_move = base_path / "movement" / f"sub_{subject_str}"
        sub_dir_rest = base_path / "rest" / f"sub_{subject_str}"

        # Load epochs
        fn_move_epochs = sub_dir_move / f"epochs_{subject_type}_{subject_str}_movement_{condition}.pkl"
        fn_rest_epochs = sub_dir_rest / f"epochs_{subject_type}_{subject_str}_rest_{condition}.pkl"
        if not (fn_move_epochs.exists() and fn_rest_epochs.exists()):
            logger.warning("Missing epochs for subject %s; skipping classification", subject_str)
            continue

        with fn_move_epochs.open("rb") as f:
            move_epochs = pickle.load(f)
        with fn_rest_epochs.open("rb") as f:
            rest_epochs = pickle.load(f)

        # Trim time
        time_axis = move_epochs.times
        start_idx = np.where(time_axis >= tmin)[0][0]
        end_idx = np.where(time_axis <= tmax)[0][-1] + 1
        X_move = move_epochs.get_data()[:, :, start_idx:end_idx]
        X_rest = rest_epochs.get_data()[:, :, start_idx:end_idx]

        y_move = np.ones(X_move.shape[0])
        y_rest = np.zeros(X_rest.shape[0])
        X_all = np.concatenate([X_move, X_rest])
        y_all = np.concatenate([y_move, y_rest])

        if analysis_type == "beta_analysis":
            all_run_scores, all_run_aucs = [], []
            for run in range(n_runs):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
                scores, aucs = [], []

                for train_idx, test_idx in skf.split(X_all, y_all):
                    X_train, X_test = X_all[train_idx], X_all[test_idx]
                    y_train, y_test = y_all[train_idx], y_all[test_idx]

                    X_train_bands, X_test_bands = [], []
                    for fmin, fmax in bands.values():
                        Xf_train = np.array([filter_data(tr, sfreq, fmin, fmax, verbose=False) for tr in X_train])
                        Xf_test = np.array([filter_data(tr, sfreq, fmin, fmax, verbose=False) for tr in X_test])
                        csp = CSP(n_components=4, log=True)
                        X_train_bands.append(csp.fit_transform(Xf_train, y_train))
                        X_test_bands.append(csp.transform(Xf_test))

                    X_train_concat = np.concatenate(X_train_bands, axis=1)
                    X_test_concat = np.concatenate(X_test_bands, axis=1)
                    clf = LDA().fit(X_train_concat, y_train)
                    scores.append(clf.score(X_test_concat, y_test))
                    try:
                        aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test_concat)[:, 1]))
                    except ValueError:
                        aucs.append(np.nan)

                all_run_scores.append(np.mean(scores))
                all_run_aucs.append(np.mean(aucs))

            subject_scores.append(np.mean(all_run_scores))
            std_scores.append(np.std(all_run_scores))
            subject_aucs.append(np.mean(all_run_aucs))
            auc_stds.append(np.std(all_run_aucs))
            logger.info("%s: Accuracy=%.3f, AUC=%.3f", subject_str, np.mean(all_run_scores), np.mean(all_run_aucs))
        else:
            logger.info("analysis_type '%s' not implemented in this refactor; skipping classification for subject %s", analysis_type, subject_str)

    # Save results and trace artifacts
    out_fn = f"analysis_results_{subject_type}.npz"
    np.savez(
        out_fn,
        subject_scores=np.array(subject_scores),
        subject_aucs=np.array(subject_aucs),
        std_scores=np.array(std_scores),
        auc_stds=np.array(auc_stds),
        sizes_per_subject=np.array(sizes_per_subject),
        top_axes=np.array(top_axes),
    )
    logger.info("Results saved to %s (includes sizes_per_subject and top_axes)", out_fn)

    return subject_scores, processed_subjects, std_scores, subject_aucs, auc_stds


if __name__ == "__main__":
    try:
        scores, subjects, stds, aucs, auc_stds = run_analysis("Patient", "beta_analysis", config_path="config.json", random_seed=0)
        logger.info("✅ Analysis completed")
    except Exception as e:
        logger.exception("Analysis failed: %s", e)