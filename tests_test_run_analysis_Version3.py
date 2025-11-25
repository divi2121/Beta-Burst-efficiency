import tempfile
import os
import numpy as np
import pytest
from pathlib import Path
import json

# Import the functions to test from run_analysis.py
import run_analysis as ra


def test_validate_config_accepts_minimal_valid_config(tmp_path):
    cfg = {
        "paths": {"preprocessed_dir": "preproc", "condition": "cond", "decim_dir": "decim"},
        "eeg": {"sfreq": 250.0, "time_window": [2.0, 10.0]},
        "analysis": {"pca_components": 5, "pca_bins": 4, "n_splits": 2, "n_runs": 1, "bands": {"b1": [15, 17]}},
        "subjects": {"patient_range": [1, 3]},
    }
    # Should not raise
    validated = ra.validate_config(cfg)
    assert validated is cfg


def test_build_global_class_indices_basic():
    # sizes_per_subject for two subjects:
    # subject 1: (2 c3_move, 1 c5_move, 0 c3_rest, 1 c5_rest)
    # subject 2: (1, 1, 1, 0)
    sizes = [(2, 1, 0, 1), (1, 1, 1, 0)]
    idx_c3_move, idx_c5_move, idx_c3_rest, idx_c5_rest = ra.build_global_class_indices(sizes)
    # Compute expected indices manually:
    # stacking order per subject: [c3_move(2), c5_move(1), c3_rest(0), c5_rest(1)] -> subject1 rows 0..3
    # subject1 indices: c3_move: [0,1], c5_move: [2], c3_rest: [], c5_rest: [3]
    # subject2 starts at offset 4: c3_move: [4], c5_move: [5], c3_rest: [6], c5_rest: []
    assert np.array_equal(idx_c3_move, np.array([0, 1, 4]))
    assert np.array_equal(idx_c5_move, np.array([2, 5]))
    assert np.array_equal(idx_c3_rest, np.array([6]))
    assert np.array_equal(idx_c5_rest, np.array([3]))


def test_validate_config_rejects_bad_config():
    bad_cfg = {"paths": {}, "eeg": {"sfreq": "not_a_number"}, "analysis": {}, "subjects": {}}
    with pytest.raises(ValueError):
        ra.validate_config(bad_cfg)