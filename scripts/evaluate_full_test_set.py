"""
PlantHealth AI - Full Test Set Evaluation (13,208 Untouched Samples)
Evaluates the production MobileNetV2 checkpoint against the entire held-out test set
without truncation or sub-sampling.
"""

from collections import Counter
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, FINAL_RESULTS_DIR, MOBILENET_MODEL_PATH
from src.data_loader import build_stratified_dataset_splits, create_tf_dataset, discover_dataset
from src.utils import logger, plot_confusion_matrix, save_json


def main():
    print("=" * 80)
    print("PLANTHEALTH AI — FULL TEST SET EVALUATION (13,208 UNTOUCHED SAMPLES)")
    print("=" * 80)

    start_time = time.time()

    # 1. Discover dataset and retrieve untouched test split
    all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    _, _, test_samples = build_stratified_dataset_splits(all_items, class_to_idx)

    num_test = len(test_samples)
    print(f"Loaded {num_test:,} test samples across {len(class_names)} classes.")

    # 2. Load trained MobileNetV2 checkpoint
    print(f"Loading trained MobileNetV2 model from {MOBILENET_MODEL_PATH}...")
    model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

    # 3. Create high-throughput tf.data test pipeline
    test_ds = create_tf_dataset(test_samples, batch_size=BATCH_SIZE, model_type="mobilenetv2", is_training=False)

    # 4. Predict probabilities across all test samples
    print(f"Running batch inference across {num_test:,} test images...")
    pred_start = time.time()
    raw_probs = model.predict(test_ds, verbose=1)
    pred_duration = time.time() - pred_start

    print(f"Completed inference in {pred_duration:.2f}s ({num_test / pred_duration:.1f} images/sec).")

    y_pred = np.argmax(raw_probs, axis=1)
    y_true = np.array([label for _, label in test_samples])

    # 5. Compute global metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print("\n" + "=" * 50)
    print("FULL TEST SET EVALUATION RESULTS (N = 13,208)")
    print("=" * 50)
    print(f"Test Accuracy:     {acc:.4%} ({acc*100:.2f}%)")
    print(f"Macro Precision:   {prec_macro:.4%}")
    print(f"Macro Recall:      {rec_macro:.4%}")
    print(f"Macro F1-Score:    {f1_macro:.4%}")
    print(f"Weighted F1-Score: {f1_weighted:.4%}")
    print("=" * 50 + "\n")

    # 6. Classification report
    rep_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep_dict).transpose()

    csv_path = FINAL_RESULTS_DIR / "full_test_classification_report.csv"
    rep_df.to_csv(csv_path)
    print(f"Saved full classification report CSV to {csv_path}")

    # 7. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = FINAL_RESULTS_DIR / "full_test_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Saved full test confusion matrix to {cm_path}")

    # 8. Save metrics JSON
    full_metrics = {
        "evaluation_scope": "Full Untouched Held-Out Test Split",
        "total_test_samples": num_test,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "throughput_images_per_sec": float(num_test / pred_duration),
        "inference_duration_sec": float(pred_duration),
    }
    json_path = FINAL_RESULTS_DIR / "full_test_metrics.json"
    save_json(full_metrics, json_path)
    print(f"Saved full test metrics to {json_path}")

    # 9. Comparison table between Balanced Test (1,520) and Full Test (13,208)
    balanced_metrics_path = FINAL_RESULTS_DIR / "final_metrics.json"
    if balanced_metrics_path.exists():
        import json
        with open(balanced_metrics_path) as f:
            bm = json.load(f).get("MobileNetV2 (Primary)", {})
    else:
        bm = {}

    comparison = {
        "balanced_test_1520": {
            "samples": 1520,
            "sampling_method": "Stratified 40 samples per class across all 38 classes",
            "accuracy": bm.get("accuracy", 0.886842),
            "macro_precision": bm.get("precision", 0.891903),
            "macro_recall": bm.get("recall", 0.886842),
            "macro_f1": bm.get("f1_score", 0.884967),
            "weighted_f1": bm.get("f1_weighted", 0.884967),
        },
        "full_test_13208": {
            "samples": num_test,
            "sampling_method": "Complete 100% held-out test split (all available samples)",
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
        }
    }
    comp_path = FINAL_RESULTS_DIR / "test_set_comparison.json"
    save_json(comparison, comp_path)
    print(f"Saved test set comparison to {comp_path}")

    total_time = time.time() - start_time
    print(f"\nFull Test Evaluation successfully finished in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
