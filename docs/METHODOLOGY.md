# PlantHealth AI – Methodology & Experimental Design

This document details the scientific methodology, validation discipline, evaluation metrics, and implementation workflow for **PlantHealth AI**.

---

## 1. Experimental Workflow

```
Raw PlantVillage Images (87,867 samples across 38 classes)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
Corrupted Image Filter                Root Leaf UUID Hashing
(Byte validation, EXIF cleanup)       (Guarantees Anti-Leakage Partition)
       │                                       │
       └───────────────────┬───────────────────┘
                           │
            Stratified Anti-Leakage Splitting
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
Training Split       Validation Split       Test Split
  (70% samples)        (15% samples)       (15% samples)
      │                    │                    │
Runtime Augmentation    No Augmentation      No Augmentation
(Rot, Flip, Zoom, Shift)   Standard Eval        Standard Eval
      │                    │                    │
      ▼                    ▼                    │
Model Training & Warmup ───┘                    │
(Checkpoints on val_accuracy)                   │
      │                                         │
      └────────────────────┬────────────────────┘
                           ▼
              Final Test Set Evaluation
      (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

---

## 2. Integrity and Anti-Leakage Protocol

1. **Independent Evaluation Guarantee:**
   The test split is never touched during hyperparameter tuning, model compilation, architecture selection, or early stopping. It represents completely held-out physical leaf specimens.
2. **Root Sample Clustering:**
   Many public datasets create synthetic variation by rotating, flipping, or zooming existing images and storing them under slightly modified names. A naive split that randomly distributes files would place rotated copies of test images in the training set, causing artificial metric inflation. PlantHealth AI extracts the root specimen ID and clusters all variations into identical splits.
3. **Reproducibility:**
   All pseudo-random operations (splitting, tensor shuffling, data augmentation) utilize fixed deterministic seeds (`seed = 42`).

---

## 3. Evaluation Metrics

- **Accuracy:** Overall proportion of correctly diagnosed leaf specimens:
  $$\text{Accuracy} = \frac{\sum_{i=1}^C \text{TP}_i}{\sum_{i=1}^C (\text{TP}_i + \text{FP}_i + \text{FN}_i + \text{TN}_i)}$$
- **Macro Precision:** Unweighted mean of class-specific precision scores, treating rare diseases with equal importance:
  $$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$$
- **Macro Recall:** Sensitivity to true foliar infections across every class:
  $$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$
- **Macro F1-Score:** Harmonic mean of macro precision and macro recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix:** $38 \times 38$ pairwise matrix capturing false positive / false negative confusions between visually analogous foliar symptoms.
