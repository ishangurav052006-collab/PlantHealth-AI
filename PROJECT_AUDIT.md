# PlantHealth AI – Comprehensive Project Audit

**Date:** 2026-10-31  
**Auditor:** Senior ML Engineer & CV Architect  
**Project:** PlantHealth AI – Plant Disease Classification using CNNs  
**Target Architecture:** Custom CNN & MobileNetV2 Transfer Learning (with VGG16 experimental benchmarks)

---

## 1. Executive Summary

This audit assesses the initial state of the `PlantHealth-AI` repository, analyzes the execution environment, verifies dataset availability, identifies gaps, and outlines an end-to-end implementation roadmap.

---

## 2. Repository & Workspace State

| Component | Status | Details |
| :--- | :--- | :--- |
| **Directory Structure** | Empty initially | Repository root `/Users/manthanshah/Documents/PlantHealth-AI` had no source files or git history. |
| **Version Control** | Not initialized | Initialized git repository; configured `.gitignore` to prevent committing virtual environments and large binary datasets. |
| **Source Code (`src/`)** | Missing | No data loaders, model builders, training loops, evaluators, or inference scripts. |
| **Trained Models (`models/`)** | Missing | No `.keras` checkpoints, weights, or `class_indices.json` mappings existed. |
| **Evaluation Artifacts (`results/`)** | Missing | No confusion matrices, classification reports, or prediction sample visualizations. |
| **Notebooks (`notebooks/`)** | Missing | No interactive prototyping or walkthrough notebooks existed. |
| **Frontend Application (`app.py`)** | Missing | No Streamlit AgTech interface existed. |
| **Test Suite (`tests/`)** | Missing | No unit tests for preprocessing, data loading, or model inference. |

---

## 3. Environment & Hardware Audit

- **Operating System:** macOS (Darwin 27.0.0, ARM64 Apple Silicon)
- **CPU / Compute Resources:** 10 CPU cores, 16 GB Unified Memory.
- **Python Runtime:**
  - Base Homebrew Python was 3.14.7 (incompatible with official TensorFlow wheels).
  - Clean Python 3.11 (`3.11.18`) installed via Homebrew.
  - Dedicated virtual environment created at `.venv` with isolated packages.
- **Dependency Stack:**
  - `tensorflow` (Keras 3 / TF 2.x)
  - `numpy`, `pandas`, `scikit-learn`
  - `matplotlib`, `seaborn`, `pillow`, `opencv-python`
  - `streamlit`
  - `pytest`, `jupyter`, `ipykernel`

---

## 4. Dataset Discovery & Integrity

- **Discovery:** Located official PlantVillage archive on the local system at `/Users/manthanshah/Downloads/archive.zip` (2.89 GB).
- **Class Coverage:** Contains all **38 canonical plant disease and healthy classes** spanning Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- **Data Splitting & Leakage Prevention:**
  - The raw dataset contains original and augmented images.
  - **Leakage Prevention Strategy:** To guarantee strict experimental integrity, dataset splitting will be stratified (70% train, 15% validation, 15% test) with a deterministic random seed (`seed=42`).
  - Augmented variations from the same source image will not cross split boundaries.
  - Image resizing standardized to $224 \times 224 \times 3$.
  - Dynamic runtime augmentation applied **strictly to the training split**, leaving validation and test splits untouched.

---

## 5. Identified Technical Challenges & Mitigation

1. **Hardware Acceleration on Apple Silicon:**
   - Model training on CPU / MPS requires optimal batch sizes ($32$) and TensorFlow `tf.data` pipeline optimizations (`cache()`, `prefetch(tf.data.AUTOTUNE)`).
2. **Preprocessing Discrepancies:**
   - Custom CNN expects normalized pixel values in $[0, 1]$.
   - MobileNetV2 uses standard `tf.keras.applications.mobilenet_v2.preprocess_input` (scaling to $[-1, 1]$).
   - The prediction pipeline must apply model-specific preprocessing deterministically.
3. **Class Mapping Consistency:**
   - Class indices must be saved to `models/class_indices.json` during data indexing and loaded identically during inference to prevent label permutations.
4. **App Robustness:**
   - Streamlit application must cleanly handle non-existent models, display actionable error states, provide leaf upload validation, and present disease mitigation advice.

---

## 6. Implementation Plan & Deliverables

### Phase 1: Environment & Project Foundation
- [x] Python 3.11 environment setup.
- [x] Git initialization & `.gitignore`.
- [ ] Directory layout setup: `src/`, `data/`, `models/`, `results/`, `notebooks/`, `tests/`, `docs/`.
- [ ] Requirements specification: `requirements.txt` and `config.py`.

### Phase 2: Data Pipeline & Preprocessing
- [ ] Dataset extraction and curation script: `src/data_loader.py`.
- [ ] Image validation, integrity checking, and stratified split generator.
- [ ] Runtime augmentation pipeline (`src/augmentation.py`) for training set only.
- [ ] Visualizations: class distribution, dataset split breakdown, augmentation comparison.

### Phase 3: Model Architecture & Training
- [ ] Custom CNN architecture (`src/custom_cnn.py`): Conv2D, BatchNorm, MaxPool, Dropout, GAP/Dense, Softmax.
- [ ] MobileNetV2 transfer learning (`src/transfer_learning.py`): ImageNet weights, classification head, two-phase fine-tuning.
- [ ] VGG16 benchmark architecture.
- [ ] Unified training runner (`src/train.py`) with `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.
- [ ] Save trained models to `models/custom_cnn.keras` and `models/plantdiseasemodel.keras`.

### Phase 4: Rigorous Evaluation
- [ ] Evaluation suite (`src/evaluate.py`): Accuracy, precision, recall, macro/weighted F1.
- [ ] Per-class classification report (`results/classification_report.csv`).
- [ ] High-resolution confusion matrix (`results/confusion_matrix.png`).
- [ ] Training history curves (`results/training_history.png`).
- [ ] Model comparison bar plots (`results/model_comparison.png`).
- [ ] 10-sample qualitative test prediction grid (`results/prediction_samples.png`).
- [ ] Comprehensive markdown evaluation report (`results/evaluation_report.md`).

### Phase 5: Inference Engine & Streamlit Application
- [ ] Reusable prediction module (`src/predict.py`) supporting top-3 confidence scores, disease taxonomy parser, and treatment guidance.
- [ ] AgTech Streamlit Dashboard (`app.py`) with 4 distinct sections: Home, Disease Detection, Model Performance, and Architecture/About.

### Phase 6: Notebooks, Unit Tests & Documentation
- [ ] `notebooks/01CustomCNN.ipynb` & `notebooks/02TransferLearning.ipynb`.
- [ ] Automated test suite (`tests/`): Data validation, image resizing, preprocessing, prediction output dimensions, confidence validity, and app startup.
- [ ] Complete documentation: `README.md`, `docs/MODEL_ARCHITECTURE.md`, `docs/DATASET.md`, `docs/METHODOLOGY.md`.
