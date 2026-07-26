# Beta-Burst-efficiency

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNE](https://img.shields.io/badge/MNE-Python-orange.svg)](https://mne.tools/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-red.svg)](LICENSE)

Preprocessing and classification code from an MS thesis project testing whether a published
beta-burst decoding method transfers from healthy motor imagery to a clinical population it had
not been validated on.

The method — detecting transient beta bursts and using their waveform shape as classification
features, rather than sustained band power — was developed by Papadopoulos et al. (2024) and
validated on healthy participants. This project applied it to a cohort of patients with
tetraplegia, locked-in syndrome, and disorders of consciousness, and benchmarked it against
conventional beta-band filtering on the same data.

**The short version of the result: the burst advantage did not transfer.** It reproduced in
healthy controls and disappeared in patients.

---

## ⚠ Repository status

**This code does not currently run standalone.** Several modules import helper functions that live
in the upstream [`bebopbci`](https://gitlab.com/sotpapad/bebopbci) repository and are not vendored
here (`time_res_features`, `plot_burst_features`, `plot_tf_features`, `plot_burst_dict`,
`preprocess`, `burst_analysis`/`TfBursts`). Restructuring this repository to depend on `bebopbci`
as a package, rather than partially vendoring it, is planned.

The clinical EEG data cannot be shared (patient privacy), so the results below are not reproducible
from this repository as-is. Adapting the pipeline to a public motor imagery dataset
(BCI Competition IV-2a via MOABB) is the intended fix for that, and is not yet done.

---

## Results

Classification was movement/attempted-movement vs. rest, using channels **C3** and **C4**,
8-second epochs, 48 epochs per condition per participant.

| Population | Beta-burst AUC | Beta-power AUC | Difference |
|---|---|---|---|
| Healthy controls | **0.72** | **0.66** | Significant (paired *t*-test) |
| Patients (tetraplegia / LIS / DOC) | ~0.70 | ~0.70 | **Not significant** |

### Interpretation

1. **The burst advantage reproduced in healthy controls.** Waveform-based burst features
   outperformed conventional beta-power features, consistent with the original report.

2. **It did not transfer to patients.** In the clinical cohort the two feature sets performed
   comparably, and the difference was not significant. The burst method was not worse — it simply
   held no advantage. Reporting this is the point of the project: a method validated on healthy
   participants should not be assumed to carry over to the population it is ultimately intended for.

3. **Absolute performance is modest in both groups**, and well below what would be needed for a
   deployable assistive BCI. Classification accuracy alone is in any case a poor clinical outcome
   measure — it says little about whether a patient can actually control a device reliably over time.

Several explanations for the patient null are plausible and this dataset cannot separate them:
differences in signal quality and artifact burden, the possibility that beta burst dynamics are
genuinely altered by the underlying pathology, and the small number of epochs per participant.
The thesis discussion treats these at length.

---

## Method

```
Raw EEG
    ↓
[1] Preprocessing  (preprocess_pipeline.py)
    • Broadband filtering
    • Zapline line-noise removal (50/60 Hz)
    • Bandpass 1–45 Hz
    • Event extraction & epoching (movement / rest)
    • AutoReject for automated artifact rejection
    ↓
[2] Burst detection  (upstream, see Provenance)
    • Superlet time-frequency decomposition
    • Beta-band (15–30 Hz) burst identification at C3, C4
    ↓
[3] Features  (run_analysis.py)
    • PCA over burst waveforms, stacked across subjects
    • Selection of discriminative PCA axes
    • Comparison arm: CSP on band-filtered data (15–17, 17–19, 19–22 Hz)
    ↓
[4] Classification  (run_analysis.py)
    • Linear Discriminant Analysis
    • Stratified k-fold CV, repeated runs
    • Scored by accuracy and ROC-AUC
```

Note that **ICA was not used**. Artifact handling was zapline + bandpass + AutoReject. ICA and ASR
were considered and left out: both can suppress the transient high-frequency content that burst
detection depends on, which is a genuine tradeoff rather than an oversight.

---

## My contribution

This repository is a mix of my own code and code from the upstream project (see Provenance). To be
explicit about the split — the burst detection methodology is not mine, and by volume most of the
code here is not mine either.

**Mine:**

| File | What it does |
|---|---|
| `preprocess_pipeline.py` | EEG preprocessing and epoching: filtering, zapline, event extraction, AutoReject integration, channel selection, CLI for batch processing over subjects and conditions |
| `run_analysis.py` | Multi-subject aggregation and alignment, per-subject scaling, PCA over stacked burst waveforms, discriminative-axis selection, the dual burst-vs-filter classification pipelines, cross-validation, results serialisation |
| `tests_run_analysis.py` | Unit tests for config validation and index computation |

In substance, my work was building the preprocessing and multi-subject analysis pipeline required
to move a published method onto a new clinical cohort, and benchmarking it honestly against the
conventional alternative. I reproduced and validated the burst method; I did not develop it.

---

## Provenance and attribution

**Burst detection and feature extraction** come from
[`bebopbci`](https://gitlab.com/sotpapad/bebopbci) by Sotiris Papadopoulos, released under
GPL-3.0-or-later. The following files in this repository are redistributed from that project:
`burst_detection.py`, `burst_features.py`, `burst_space.py`, `burst_modeling.py`,
`classification_pipelines.py`, `help_funcs.py`, `lagged_coherence.py`, `plot_tf_activity.py`,
`zapline_iter.py`. Original author docstrings are intact.

*Modifications: none. These files are redistributed unmodified — verified byte-identical to
upstream at commit `6362c8c` (2025-06-09).*

`bebopbci` in turn builds on:
- DANC lab burst detection — https://github.com/danclab/burst_detection
- Gregor Mönke's superlet implementation
- MNE-Python, and MOABB for open dataset access

The work was carried out at CRNL / INSERM Lyon under the supervision of Jérémie Mattout (CRNL) and
James Bonaiuto (CNRS), with Sotiris Papadopoulos, and locally supervised by Hasan Mohammad
(IISER Mohali).

---

## Repository structure

```
Beta-Burst-efficiency/
│
├── preprocess_pipeline.py       # [MINE] preprocessing & epoching
├── run_analysis.py              # [MINE] PCA, classification, validation
├── tests_run_analysis.py        # [MINE] unit tests
│
├── burst_detection.py           # [UPSTREAM] burst detection
├── burst_features.py            # [UPSTREAM] feature extraction
├── burst_space.py               # [UPSTREAM] burst space modeling
├── burst_modeling.py            # [UPSTREAM]
├── classification_pipelines.py  # [UPSTREAM] additional classifiers
├── help_funcs.py                # [UPSTREAM] utilities
├── lagged_coherence.py          # [UPSTREAM] connectivity analysis
├── plot_tf_activity.py          # [UPSTREAM] visualisation
├── zapline_iter.py              # [UPSTREAM] iterative zapline
│
├── config.json                  # analysis configuration
├── requirements.txt
└── LICENSE                      # GPL-3.0
```

---

## Usage

Subject to the status caveat above — the imports do not currently resolve without `bebopbci`.

**Preprocessing:**
```bash
python preprocess_pipeline.py --subject 1 --condition ZAP_45_BP --data-type Patient
```

**Analysis:**
```bash
python run_analysis.py
```

Or programmatically:
```python
from run_analysis import run_analysis

scores, subjects, stds, aucs, auc_stds = run_analysis(
    subject_type="Patient",
    analysis_type="beta_analysis",
    config_path="config.json",
    random_seed=42,
)
```

Configuration lives in `config.json` (included). Results are written as `.npz` containing
`subject_scores`, `subject_aucs`, `std_scores`, `sizes_per_subject`, and `top_axes`.

---

## Limitations

- **Not reproducible from this repository** — clinical data cannot be shared, and the code does not
  currently run standalone.
- **Small epoch counts.** 48 epochs per condition per participant is few for waveform-level PCA.
- **Two channels.** C3/C4 assumes a canonical sensorimotor topography. In patients with structural
  damage or long-term reorganisation that assumption may fail; a data-driven channel selection
  would be a better approach.
- **Epoch length is a tradeoff.** 8-second epochs give enough bursts per epoch for stable waveform
  statistics, at the cost of temporal precision and of any real-time applicability.
- **PCA is not obviously the right decomposition** for burst waveforms — it was adequate, not optimal.

### Planned

- [ ] Restructure to depend on `bebopbci` rather than vendoring it
- [ ] Reproduce the pipeline on BCI Competition IV-2a via MOABB, with regenerable figures
- [ ] Pin `requirements.txt` (currently unpinned, and missing `meegkit` and `pytest`)
- [ ] Add burst waveform / PCA component / confusion matrix plots

---

## References

- Papadopoulos, S., Darmet, L., Szul, M. J., Congedo, M., Bonaiuto, J. J., & Mattout, J. (2024).
  *Surfing beta burst waveforms to improve motor imagery-based BCI.*
  Imaging Neuroscience, 2, imag-2-00391.
  https://doi.org/10.1162/imag_a_00391

- Papadopoulos, S., et al. (2024). *Improved motor imagery decoding with spatiotemporal filtering
  based on beta burst kernels.* Graz BCI Conference 2024.

- Szul, M. J., Papadopoulos, S., Alavizadeh, S., Daligaut, S., Schwartz, D., Mattout, J., &
  Bonaiuto, J. J. (2023). *Diverse beta burst waveform motifs characterize movement-related
  cortical dynamics.* Progress in Neurobiology, 228, 102490.
  https://doi.org/10.1016/j.pneurobio.2023.102490

- Little, S., Bonaiuto, J., Barnes, G., & Bestmann, S. (2019). *Human motor cortical beta bursts
  relate to movement planning and response errors.* PLOS Biology, 17(10), e3000479.
  https://doi.org/10.1371/journal.pbio.3000479

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). This repository incorporates GPL-3.0-or-later code from
`bebopbci`; the combined work is distributed under the same terms.

## Contact

- GitHub: [@divi2121](https://github.com/divi2121)
