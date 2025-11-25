# Beta-Burst Efficiency — Synthetic-data friendly analysis

Overview
--------
This repository contains an EEG analysis pipeline that extracts beta-burst waveforms from motor electrodes (C3 / C5 indices in the code), performs dimensionality reduction (PCA) on waveform shapes, and runs two classification analyses:

- `beta_analysis`: classifies movement vs rest using bandpass filtering → CSP → LDA with cross-validation.
- `burst_analysis`: uses PCA-derived waveform filters (average waveforms from PCA axes) to convolve with epochs, then CSP+LDA classification.

A data generator is included so you can run the pipeline end-to-end using synthetic data if real EEG / burst files are not available.

Quick features
- Per-subject burst extraction and PCA on stacked burst waveforms
- Axis selection that highlights PCA components that differentiate C3 vs C5 differences in movement vs rest
- Two classification modes (beta-based and burst-filter-based)
- Synthetic data generator to produce dummy bursts and Epochs pickles for testing

Repository layout (relevant files)
- generate_synthetic_data.py — create synthetic burst .npy files and pickled MNE Epochs
- run_analysis.py (or your analysis script) — main pipeline: burst collection, PCA, classification
- config.json — configuration (paths, sampling freq, PCA and analysis params)
- results: produced .npz summary files

Requirements
------------
- Python 3.8+
- numpy, scipy, pandas, scikit-learn, mne, matplotlib
- If using conda:
  conda create -n betaenv python=3.9 numpy scipy pandas scikit-learn mne matplotlib
- Or pip:
  pip install numpy scipy pandas scikit-learn mne matplotlib

Configuration
-------------
The pipeline reads `config.json`. Minimal example (adapt paths to your environment):

```json
{
  "paths": {
    "preprocessed_dir": "preprocessed",
    "condition": "cond",
    "decim_dir": "decim_4"
  },
  "eeg": {
    "sfreq": 250.0,
    "time_window": [2.0, 10.0]
  },
  "analysis": {
    "pca_components": 10,
    "pca_bins": 7,
    "excluded_axes": [],
    "n_splits": 2,
    "n_runs": 5,
    "bands": {
      "band_1": [15, 17],
      "band_2": [17, 19],
      "band_3": [19, 22]
    }
  },
  "subjects": {
    "patient_range": [1, 55],
    "patient_exclude": [9,10],
    "subject_range": [1, 30]
  }
}
```

Generating synthetic data (recommended when you do not have real EEG / burst files)
----------------------------------------------------------------
A helper script (`generate_synthetic_data.py`) creates:
- per-subject folders:
  - `{preprocessed_dir}/{condition}/{decim}/{subject_type}_data/movement/sub_{XX}/beta_bursts_superlets_nfs.npy`
  - same for `rest/`
- per-subject epoch pickles:
  - `epochs_{subject_type}_{XX}_movement_{condition}.pkl` and rest equivalent

Usage:
- Edit `config.json` or set parameters at top of `generate_synthetic_data.py`.
- Run:
  python generate_synthetic_data.py

This produces data consistent with the filenames expected by the analysis script and is deterministic when seed is set.

Running the analysis
--------------------
Example:
1. Generate synthetic data (if needed):
   python generate_synthetic_data.py

2. Run analysis (change subject_type and mode as needed):
   python run_analysis.py
   or call the function:
   from run_analysis import run_analysis
   run_analysis("Patient", "beta_analysis")

Outputs
-------
- analysis_results_Patient.npz (example) containing arrays:
  - subject_scores (accuracy), subject_aucs (AUC), std_scores, auc_stds
- Console logs show top PCA axes and per-subject performance.

Notes & best practices
----------------------
- File/format expectations:
  - Burst files must be loadable via np.load(..., allow_pickle=True) and provide data[3] and data[5] entries with fields:
    - 'trial' (1D int array), 'waveform' (2D array n_bursts x n_timepoints), 'peak_time' (1D float array)
  - Epoch pickles must be MNE Epochs-like objects with `.times` and `.get_data()` (EpochsArray is fine).
- Ensure `times` and `sfreq` are consistent when constructing synthetic epochs.
- Use a fixed random seed for reproducibility (both generator and analysis accept/expect seeding).
- CSP and StratifiedKFold require sufficient trials per class; add guards for small-sample cases.
- If you want global normalization for PCA, fit a single scaler across subjects instead of per-subject scaling.

Contact / getting help
----------------------
If you want me to:
- add the README file to the repo,
- create a small unit test that runs the pipeline on synthetic data,
- or refactor run_analysis into smaller testable functions,
tell me which and I’ll prepare a patch or a PR.
