# PlantHealth AI – Model Performance Diagnostic Report

**Date:** 2026-10-31  
**Investigator:** Senior ML Engineer & CV Architect  
**Objective:** Root cause analysis of initial low test accuracy (25.20% MobileNetV2, 0.00% Custom CNN, 600-slice 10.33%) and establishment of corrective actions.

---

## 1. Executive Summary

A comprehensive forensic audit of the codebase, training artifacts, and data pipelines revealed **seven critical root causes** that severely undermined model accuracy in the previous execution. These issues explain both the low test accuracy on MobileNetV2 and the extreme discrepancies in test evaluations.

---

## 2. Forensic Findings & Root Cause Analysis

### Root Cause 1: Preprocessing & Augmentation Order Mismatch (`RandomBrightness` Value Range Bug)
- **The Bug:** In `src/data_loader.py`, `create_tf_dataset` mapped `_parse_and_preprocess` first, which scaled images to `[-1, 1]` via `mobilenet_v2.preprocess_input`. Immediately afterward, `aug_layer = get_training_augmentation_pipeline()` was applied on the already-normalized dataset.
- **Impact:** In `src/augmentation.py`, `layers.RandomBrightness(factor=0.15, value_range=(0, 255))` explicitly specifies `value_range=(0, 255)`. When Keras executes `RandomBrightness` with `value_range=(0, 255)` on tensor inputs that are already in `[-1, 1]`, it clips all negative values to zero and shifts values by hundreds of units! 
- **Verification:** Empirically verified in isolated Python test: an input of `[-1.0, 0.0, 1.0]` was clipped to `[[[0.0, 0.0, 0.0]]]`. Consequently, training images presented severely distorted, blacked-out, or clipped artifacts to the network, crippling representation learning.

### Root Cause 2: Broken Sample ID Extraction on Class 08 (`Corn Common Rust`)
- **The Bug:** PlantVillage filenames for `Corn_(maize)___Common_rust_` do not contain the `___` separator present in all other 37 classes. Instead, files follow the pattern `RS_Rust <number>.JPG` and `RS_Rust <number>_flipLR.JPG`.
- **Impact:** In `_extract_sample_id`, the fallback logic executed `filename.split('.')[0].split('_')[0]`. For all 2,384 files in this class, this returned the single string `"RS"`.
- **Result:** The stratified split algorithm saw exactly $N = 1$ unique sample for Class 08. Because $0.70 \times 1 = 1$, all 2,384 images went into the training set, while **Validation received 0 images and Testing received 0 images**! In the previous classification report, Class 08 had literally **Support = 0** and **F1 = 0.00%**.

### Root Cause 3: 600-Sample Slice Evaluation Discrepancy
- **The Bug:** In `src/evaluate.py`, when `--max-test-samples 600` was specified, the script executed `test_samples = test_samples[:600]`.
- **Impact:** Taking the first 600 elements of an arbitrarily shuffled list resulted in non-uniform class sampling, with some classes receiving 28 samples, others 0 (Class 08 had 0), creating severe distortion.
- **Resolution:** Stratified sampling must be enforced across all evaluation modes, ensuring proportional representation of all 38 classes.

### Root Cause 4: Two-Stage Transfer Learning Checkpoint Overwrite
- **The Bug:** In `src/train.py`, `train_mobilenetv2` instantiated `ModelCheckpoint` separately in Stage 1 (Warmup) and Stage 2 (Fine-Tuning), pointing to the exact same file path (`models/plantdiseasemodel.keras`).
- **Impact:** When Stage 2 began, the new `ModelCheckpoint` instance reset its internal `best` metric to `-inf`. At Epoch 5 (the first epoch of fine-tuning), `val_accuracy` was saved regardless of whether it was better than Stage 1's best checkpoint. Furthermore, `model.evaluate` at the end evaluated the in-memory model rather than the best restored weights.

### Root Cause 5: Under-Tuned Warmup Learning Rate & Premature Termination
- **The Problem:** The classification head contains 337,702 randomly initialized parameters. The warmup learning rate was set to $\eta = 10^{-4}$ for only 4 epochs. The loss at Epoch 4 was still hovering at $2.70$ (random initialization starts around $5.1$). The head had not yet converged before the learning rate was dropped by $10\times$ to $10^{-5}$ for fine-tuning.
- **Correct Configuration:** Head warmup requires $\eta = 10^{-3}$ for sufficient gradient descent, followed by fine-tuning at $\eta = 10^{-5}$ once the head is stable.

### Root Cause 6: Custom CNN Single-Epoch Interruption
- **The Problem:** Custom CNN training was previously aborted after 1 single epoch due to slow CPU execution with high batch counts. While training accuracy on that single memorized batch subset reached 70%, its test accuracy on the full 38-class held-out test set was $0.00\%$.
- **Correct Configuration:** Proper optimization (depthwise separable convolutions, batch size 32, caching, and 10–15 complete epochs) is required for genuine convergence.

### Root Cause 7: Normalization Architecture
- **Solution:** Embedding `mobilenet_v2.preprocess_input` directly as the first functional layer inside the model guarantees that any image input in `[0, 255]` (during training, validation, testing, or Streamlit inference) is processed identically without relying on external preprocessing calls.

---

## 3. Systematic Corrective Action Plan

1. **Fix `src/data_loader.py`:**
   - Update `_extract_sample_id` with regex cleaning to correctly parse `RS_Rust <number>` for Corn Common Rust, yielding 1,192 unique sample IDs.
   - Reorder `create_tf_dataset` so augmentation operates exclusively on raw `[0, 255]` images.
2. **Fix `src/transfer_learning.py`:**
   - Embed `preprocess_input` inside model architecture.
   - Clean classification head: `GAP -> Dropout(0.3) -> Dense(256, relu) -> Dropout(0.3) -> Dense(38, softmax)`.
   - Unified checkpoint manager that tracks the global best `val_accuracy` across both warmup and fine-tuning stages.
3. **Fix `src/train.py`:**
   - Set warmup learning rate to $10^{-3}$, fine-tuning learning rate to $10^{-5}$.
   - Ensure `restore_best_weights=True` so evaluations assess the optimal model.
4. **Fix `src/evaluate.py`:**
   - Replace naive slice truncation with stratified subset sampling when capped evaluation is requested.
   - Save all final artifacts to `results/final/`.
5. **Retrain Both Models Sequentially on Real Data.**
