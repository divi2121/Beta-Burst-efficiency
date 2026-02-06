<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6cb25ca0-3a0a-4521-9db8-accf7564c9dc" /># Beta-Burst Analysis for Motor Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNE](https://img.shields.io/badge/MNE-Python-orange.svg)](https://mne.tools/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-red.svg)](LICENSE)

A novel EEG analysis pipeline comparing **beta-burst** detection versus traditional **beta-filtering** approaches for classifying motor intention in healthy subjects and motor-impaired patients. This research demonstrates that beta bursts serve as more accurate neurological markers for motor classification tasks.

---

## Key Findings

- **Healthy Subjects**: 78% classification accuracy using beta-burst analysis
- **Patient Population**: 70% classification accuracy 
- **Clinical Significance**: Beta-burst approach outperforms traditional beta-filtering methods, suggesting superior neurological markers for motor intention
- **Application**: Potential improvements for brain-computer interfaces (BCIs) and motor recovery prediction systems

---

## Research Context

### Background
This project implements and validates cutting-edge neuroscience methods for analyzing motor-related brain activity using electroencephalography (EEG). The work focuses on:

- **Motor Imagery** (healthy subjects): Imagining hand movements without execution
- **Motor Intention** (patient population): Attempting but unable to execute movements due to motor impairment
- **Classification Task**: Distinguishing motor activity from rest states

### Collaboration
This repository contains a research implementation developed in collaboration with a PhD candidate who authored the core burst detection algorithms. My contributions focus on the preprocessing pipeline, analysis framework, and classification validation.

### Clinical Relevance
Understanding beta-burst dynamics has implications for:
- Stroke rehabilitation and recovery prediction
- Brain-computer interface development for assistive devices
- Non-invasive assessment of motor system function
- Real-time neurofeedback systems

---

## Technical Approach

### Pipeline Overview

```
Raw EEG Data
    ↓
[1] Preprocessing Pipeline
    • Bandpass filtering (1-45 Hz)
    • Line noise removal (zapline)
    • Event extraction & epoching
    • Artifact rejection (AutoReject)
    ↓
[2] Burst Detection
    • Beta-band (15-30 Hz) burst identification
    • Waveform extraction from motor electrodes (C3, C5)
    ↓
[3] Feature Extraction
    • PCA dimensionality reduction on burst waveforms
    • Common Spatial Patterns (CSP) for spatial filtering
    ↓
[4] Classification
    • Linear Discriminant Analysis (LDA)
    • Stratified k-fold cross-validation
    • Comparison: Beta-burst vs Beta-filter approaches
```

### Methods Comparison

| Approach | Description | Result |
|----------|-------------|---------|
| **Beta-Burst Pipeline** | Detects transient bursts, extracts waveform features, applies PCA+CSP | **78%** (healthy), **70%** (patients) |
| **Beta-Filter Pipeline** | Traditional bandpass filtering, CSP spatial filtering | Lower accuracy |

---

## My Contributions

### 1. EEG Preprocessing Pipeline (`preprocess_pipeline.py`)
**300 lines | Signal Processing & Data Preparation**

**Key Components:**
- **Signal Cleaning**: 
  - Initial broadband filtering (0-120 Hz)
  - Zapline-based line noise removal (50/60 Hz)
  - Final bandpass filtering (1-45 Hz)
  
- **Event Processing**:
  - Automated event extraction from EEG annotations
  - Separate epoching for movement and rest conditions
  - Configurable time windows for different experimental conditions
  
- **Quality Control**:
  - AutoReject integration for automated artifact detection
  - Channel selection (motor cortex focus: C3, C4, Cz, etc.)
  - Threshold computation for bad epoch rejection

- **Modular Architecture**:
  - Command-line interface for batch processing
  - Configurable preprocessing conditions
  - Logging system for reproducibility

**Technical Skills Demonstrated:**
- MNE-Python for EEG data handling
- Digital signal processing (filtering, artifact removal)
- Experimental design understanding (epoching strategies)
- Production-ready code structure (CLI, logging, error handling)

---

### 2. Classification Analysis (`run_analysis.py`)
**423 lines | Machine Learning & Statistical Analysis**

**Key Components:**

- **Data Aggregation**:
  - Multi-subject burst collection and alignment
  - Trial randomization and stratification
  - Per-subject data normalization (RobustScaler)

- **Dimensionality Reduction**:
  - PCA on stacked burst waveforms across subjects
  - Automated selection of discriminative PCA axes
  - Class-specific mean computation (C3 vs C5, movement vs rest)

- **Classification Framework**:
  - Dual-pipeline implementation (beta-burst vs beta-filter)
  - Multi-band filtering (15-17 Hz, 17-19 Hz, 19-22 Hz)
  - Common Spatial Patterns (CSP) for spatial feature extraction
  - Linear Discriminant Analysis for classification
  - Stratified k-fold cross-validation with multiple runs

- **Validation & Robustness**:
  - Config validation with informative error messages
  - Proper handling of class indices for multi-subject PCA
  - ROC-AUC scoring in addition to accuracy
  - Results serialization with metadata (`.npz` format)

**Technical Skills Demonstrated:**
- scikit-learn ML pipeline design
- Cross-validation best practices
- Handling high-dimensional EEG data
- Statistical validation (multiple runs, stratification)
- Code refactoring for correctness and maintainability

---

### 3. Testing & Validation (`tests_run_analysis.py`)
**43 lines | Unit Testing**

**Key Components:**
- Config validation tests
- Index computation verification
- Input validation for edge cases

**Technical Skills Demonstrated:**
- Pytest framework usage
- Test-driven development practices
- Edge case handling

---

## Technical Stack

### Core Dependencies
```python
# Signal Processing
mne-python          # EEG data handling, CSP, filtering
scipy               # Signal processing utilities

# Machine Learning
scikit-learn        # PCA, LDA, cross-validation, metrics
numpy               # Numerical computations

# Utilities
pathlib             # Modern file path handling
logging             # Execution tracking
json                # Configuration management
```

### Analysis Methods
- **Common Spatial Patterns (CSP)**: Spatial filtering for EEG source separation
- **Principal Component Analysis (PCA)**: Dimensionality reduction on burst waveforms
- **Linear Discriminant Analysis (LDA)**: Binary classification (movement vs rest)
- **Stratified K-Fold CV**: Robust validation with class balance preservation

---

## Repository Structure

```
Beta-Burst-efficiency/
│
├── preprocess_pipeline.py          # [MY CODE] EEG preprocessing & epoching
├── run_analysis.py                 # [MY CODE] PCA, classification, validation
├── tests_test_run_analysis_Version3.py  # [MY CODE] Unit tests
│
├── burst_detection.py              # [COLLABORATOR] Burst detection algorithms
├── burst_features.py               # [COLLABORATOR] Feature extraction methods
├── burst_space.py                  # [COLLABORATOR] Burst space modeling
├── classification_pipelines.py     # [COLLABORATOR] Additional classifiers
├── help_funcs.py                   # [COLLABORATOR] Utility functions
├── lagged_coherence.py             # [COLLABORATOR] Connectivity analysis
├── plot_tf_activity.py             # [COLLABORATOR] Visualization utilities
├── zapline_iter.py                # [COLLABORATOR] Impelementing zapline artifact rejection

│
├── config.json                     # Configuration file (not included)
└── README.md                       # This file
```

---

## Usage

### Configuration
The pipeline requires a `config.json` file with the following structure:

```json
{
  "paths": {
    "preprocessed_dir": "path/to/preprocessed/data",
    "condition": "ZAP_45_BP",
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
    "n_splits": 5,
    "n_runs": 10,
    "bands": {
      "band_1": [15, 17],
      "band_2": [17, 19],
      "band_3": [19, 22]
    }
  },
  "subjects": {
    "patient_range": [1, 55],
    "patient_exclude": [9, 10],
    "subject_range": [1, 30]
  }
}
```

### Running the Pipeline

#### Preprocessing
```bash
python preprocess_pipeline.py --subject 1 --condition ZAP_45_BP --data-type Patient
```

#### Analysis
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
    random_seed=42
)
```

### Output
Results are saved as `.npz` files containing:
- `subject_scores`: Classification accuracy per subject
- `subject_aucs`: ROC-AUC scores per subject
- `std_scores`: Standard deviations across CV runs
- `sizes_per_subject`: Burst counts per class/subject
- `top_axes`: Selected PCA components

---

## Results & Interpretation

### Classification Performance

| Population | Accuracy | AUC | Interpretation |
|------------|----------|-----|----------------|
| Healthy Subjects | 78% | ~0.82 | Strong discrimination between motor imagery and rest |
| Motor-Impaired Patients | 70% | ~0.75 | Consistent classification despite motor deficits |

### Key Insights

1. **Beta Bursts are Superior Markers**: The burst-based approach consistently outperforms traditional beta-power filtering, suggesting that transient burst events carry more discriminative information than sustained oscillatory power.

2. **Cross-Population Validity**: The method generalizes from healthy motor imagery to impaired motor intention, indicating robust neural correlates despite different task demands.

3. **Spatial Specificity**: Using motor cortex electrodes (C3, C5) with CSP spatial filtering effectively isolates motor-related activity from background noise.

4. **Clinical Potential**: 70% accuracy in patients is sufficient for assistive BCI applications and could serve as an objective biomarker for motor recovery assessment.

---

##  Limitations & Future Work

### Current Limitations

1. **Data Availability**: Original clinical EEG data cannot be shared due to privacy regulations   
2. **Hardware Requirements**: Analysis requires substantial memory for multi-subject PCA (recommend 16GB+ RAM)
3. **Generalization**: Results specific to motor imagery/intention tasks with limited electrode coverage

### Planned Improvements

- [ ] **Public Dataset Adaptation**: Adapt code for BCI Competition datasets (e.g., Dataset IIa, IIb)
- [ ] **Synthetic Data Generation**: Create realistic synthetic EEG for testing and demonstration
- [ ] **Visualization Suite**: Add burst waveform plots, PCA component visualization, classification confusion matrices
- [ ] **Documentation Expansion**: Add Jupyter notebook tutorials for method explanation
- [ ] **Benchmarking**: Compare against state-of-the-art BCI methods from literature
- [ ] **Real-time Implementation**: Optimize for online BCI applications

---

##  Related Publications

This pipeline was inspired by and aligns with the findings of:

- Papadopoulos, S., Darmet, L., Szul, M. J., Congedo, M., Bonaiuto, J. J., & Mattout, J. (2024).  
  *Surfing beta burst waveforms to improve motor imagery-based BCI.*  
  Imaging Neuroscience, 2, imag-2-00391.  
  [https://doi.org/10.1162/imag_a_00391](https://doi.org/10.1162/imag_a_00391)

- Szul, M. J., Papadopoulos, S., Alavizadeh, S., Daligaut, S., Schwartz, D., Mattout, J., & Bonaiuto, J. J. (2023).  
  *Diverse beta burst waveform motifs characterize movement-related cortical dynamics.*  
  Progress in Neurobiology, 228, 102490.  
  [https://doi.org/10.1016/j.pneurobio.2023.102490](https://doi.org/10.1016/j.pneurobio.2023.102490)

- Little, S., Bonaiuto, J., Barnes, G., & Bestmann, S. (2019).  
  *Human motor cortical beta bursts relate to movement planning and response errors.*  
  PLOS Biology, 17(10), e3000479.  
  [https://doi.org/10.1371/journal.pbio.3000479](https://doi.org/10.1371/journal.pbio.3000479)

---

##  Collaborator Acknowledgments
**Burst Detection Algorithms**:Dr.James Bonaiuto CNRS (Team Leader) , Dr Jeremie Mattout Cophy and Inserm (Team Leader) and Sotiris Papadopoulos (PhD Candidate) - 
Core burst detection methodology - Feature extraction framework - Helper utilities and visualization tools

**My Role**: Preprocessing pipeline, analysis framework, classification validation, and documentatio
---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions about this implementation or potential collaborations:
- GitHub: [@divi2121](https://github.com/divi2121)
- Repository: [Beta-Burst-efficiency](https://github.com/divi2121/Beta-Burst-efficiency)

---

## Skills Demonstrated

This project showcases competencies in:

**Neuroscience & Signal Processing**
- EEG data analysis and interpretation
- Digital filtering and artifact removal
- Event-related potential analysis
- Motor cortex neurophysiology understanding

**Machine Learning & Statistics**
- Dimensionality reduction (PCA)
- Spatial filtering (CSP)
- Classification algorithms (LDA)
- Cross-validation methodologies
- Performance metrics (accuracy, AUC)

**Software Engineering**
- Modular, maintainable code architecture
- Command-line interfaces
- Configuration management
- Unit testing
- Error handling and logging
- Documentation best practices

**Research Skills**
- Literature implementation
- Experimental validation
- Collaboration on complex projects
- Scientific communication

---

*This README demonstrates a complete machine learning pipeline for biomedical signal analysis, from raw data preprocessing through statistical validation of results.*
