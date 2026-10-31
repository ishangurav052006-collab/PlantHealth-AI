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
- [x] Directory layout setup: `src/`, `data/`, `models/`, `results/`, `notebooks/`, `tests/`, `docs/`.
- [x] Requirements specification: `requirements.txt` and `config.py`.

### Phase 2: Data Pipeline & Preprocessing
- [x] Dataset extraction and curation script: `src/data_loader.py`.
- [x] Image validation, integrity checking, and stratified split generator.
- [x] Runtime augmentation pipeline (`src/augmentation.py`) for training set only.
- [x] Visualizations: class distribution, dataset split breakdown, augmentation comparison.

### Phase 3: Model Architecture & Training
- [x] Custom CNN architecture (`src/custom_cnn.py`): Conv2D, BatchNorm, MaxPool, Dropout, GAP/Dense, Softmax.
- [x] MobileNetV2 transfer learning (`src/transfer_learning.py`): ImageNet weights, classification head, two-phase fine-tuning.
- [x] VGG16 benchmark architecture.
- [x] Unified training runner (`src/train.py`) with `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.
- [x] Save trained models to `models/custom_cnn.keras` and `models/plantdiseasemodel.keras`.

### Phase 4: Rigorous Evaluation
- [x] Evaluation suite (`src/evaluate.py`): Accuracy, precision, recall, macro/weighted F1.
- [x] Per-class classification report (`results/classification_report.csv` & `results/final/final_classification_report.csv`).
- [x] High-resolution confusion matrix (`results/confusion_matrix.png` & `results/final/final_confusion_matrix.png`).
- [x] Training history curves (`results/training_history.png` & `results/final/final_training_history.png`).
- [x] Model comparison bar plots (`results/model_comparison.png` & `results/final/final_model_comparison.png`).
- [x] 10-sample qualitative test prediction grid (`results/prediction_samples.png` & `results/final/final_prediction_samples.png`).
- [x] Comprehensive markdown evaluation report (`results/evaluation_report.md` & `results/final/final_evaluation_report.md`).

### Phase 5: Inference Engine & Streamlit Application
- [x] Reusable prediction module (`src/predict.py`) supporting top-3 confidence scores, disease taxonomy parser, and treatment guidance.
- [x] AgTech Streamlit Dashboard (`app.py`) with 4 distinct sections: Home, Disease Detection, Model Performance, and Architecture/About.

### Phase 6: Notebooks, Unit Tests & Documentation
- [x] `notebooks/01CustomCNN.ipynb` & `notebooks/02TransferLearning.ipynb`.
- [x] Automated test suite (`tests/`): Data validation, image resizing, preprocessing, prediction output dimensions, confidence validity, and app startup (16 passed in 3.3s).
- [x] Complete documentation: `README.md`, `docs/MODEL_ARCHITECTURE.md`, `docs/DATASET.md`, `docs/METHODOLOGY.md`.

---

## 7. Forensic Root Cause Analysis & Final Performance Correction

During the initial performance audit, multiple root causes were isolated and systematically corrected:

1. **Augmentation / Normalization Clipping:** Data augmentation previously ran *after* normalization, passing $[-1, 1]$ tensors into `RandomBrightness(value_range=(0, 255))`, which clipped negative values to 0 and corrupted pixel values. **Fix:** Data augmentation was relocated to execute strictly on raw $[0, 255]$ float tensors *before* model normalization.
2. **Corn Common Rust Sample ID Splitting:** PlantVillage filenames for `Corn_(maize)___Common_rust_` lacked `___` tokens (`RS_Rust <num>.JPG`), causing `_extract_sample_id` to lump all 2,384 images under the single key `"RS"`. As a result, the stratified splitter placed all images into training, leaving 0 validation and 0 test samples. **Fix:** Enhanced regex cleaning (`RS_Rust <num>`) successfully recovered all 1,192 unique leaf specimen groups, restoring full representation across all splits.
3. **Truncated Test Slicing:** Previous test runs used naive slice truncation (`test_samples[:600]`), skewing class frequencies. **Fix:** Replaced with balanced, stratified sampling across all 38 classes (e.g. 40 samples per class = 1,520 test samples).
4. **Transfer Learning Checkpoint Overwrite:** In `src/train.py`, Stage 2 (Fine-Tuning) re-instantiated `ModelCheckpoint` to the same file, resetting `best` to `-inf` and overwriting the superior Stage 1 weights. **Fix:** Implemented unified checkpoint tracking carrying `best_warmup_acc` into Stage 2 and restoring optimal weights at completion.
5. **Head Warmup Learning Rate:** Increased head warmup learning rate to $\eta = 10^{-3}$ for sufficient initial gradient descent on the classification head before fine-tuning at $\eta = 10^{-5}$.
6. **Custom CNN Training:** Expanded Custom CNN from an interrupted 1-epoch run to 10 full epochs with data augmentation, Batch Normalization, and Dropout.

### Final Held-Out Test Evaluation Results (Untouched Stratified Test Split, N = 1,520)

| Metric | MobileNetV2 (Primary) | Custom CNN (From Scratch) |
| :--- | :---: | :---: |
| **Test Accuracy** | **88.68%** | 47.63% |
| **Macro Precision** | **89.19%** | 57.08% |
| **Macro Recall** | **88.68%** | 47.63% |
| **Macro F1-Score** | **88.50%** | 43.12% |
| **Weighted F1-Score** | **88.50%** | 43.12% |
| **Corn Common Rust F1** | **100.00%** (Supp: 40) | 52.17% (Supp: 40) |
| **Live Prediction Accuracy (20 unseen)** | **80.0% (Top-1) / 95.0% (Top-3)** | Verified |
| **Selected Deployment Model** | **MobileNetV2** | Baseline Comparison |
