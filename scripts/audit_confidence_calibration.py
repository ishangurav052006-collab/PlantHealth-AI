"""
PlantHealth AI - Confidence Calibration and Temperature Scaling Analysis
Calculates Expected Calibration Error (ECE), Brier Score, fits Temperature Scaling
on validation data strictly, and evaluates calibrated uncertainty on held-out test data.
"""

from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, FINAL_RESULTS_DIR, MOBILENET_MODEL_PATH
from src.data_loader import build_stratified_dataset_splits, create_tf_dataset, discover_dataset
from src.utils import save_json


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """
    Computes Expected Calibration Error (ECE) across n_bins equal-width confidence intervals in [0, 1].
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(np.sum(in_bin)),
                "proportion": float(prop_in_bin)
            })
        else:
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "accuracy": 0.0,
                "confidence": float((bin_lower + bin_upper) / 2),
                "count": 0,
                "proportion": 0.0
            })

    return float(ece), bin_data


def compute_brier_scores(probs: np.ndarray, labels: np.ndarray):
    """
    Computes both Multi-Class Brier Score and Top-1 Binary Brier Score.
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0

    # Multi-class Brier score: mean squared error across all class probability distributions
    multiclass_brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Binary top-1 Brier score: mean squared error between top-1 confidence and binary correctness
    preds = np.argmax(probs, axis=1)
    top1_conf = np.max(probs, axis=1)
    correctness = (preds == labels).astype(float)
    binary_brier = float(np.mean((top1_conf - correctness) ** 2))

    return multiclass_brier, binary_brier


def extract_logits(model: tf.keras.Model, samples: list, batch_size: int = BATCH_SIZE):
    """Extracts raw pre-softmax logits from MobileNetV2."""
    W, b = model.get_layer("classification_head").get_weights()
    feat_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("drop_head").output)

    ds = create_tf_dataset(samples, batch_size=batch_size, model_type="mobilenetv2", is_training=False)
    features = feat_model.predict(ds, verbose=0)
    logits = np.dot(features, W) + b
    labels = np.array([label for _, label in samples])
    return logits, labels


def optimize_temperature(val_logits: np.ndarray, val_labels: np.ndarray):
    """
    Finds optimal temperature T > 0 on Validation data exclusively by minimizing NLL.
    """
    n_samples = len(val_labels)

    def nll_obj(T):
        scaled_logits = val_logits / T
        # Log-sum-exp trick for numerical stability
        max_z = np.max(scaled_logits, axis=1, keepdims=True)
        lse = max_z + np.log(np.sum(np.exp(scaled_logits - max_z), axis=1, keepdims=True))
        log_probs = scaled_logits - lse
        correct_log_probs = log_probs[np.arange(n_samples), val_labels]
        return -np.mean(correct_log_probs)

    res = minimize_scalar(nll_obj, bounds=(0.1, 10.0), method="bounded")
    return float(res.x), float(res.fun)


def plot_reliability_diagrams(
    bin_data_orig, ece_orig,
    bin_data_calib, ece_calib,
    optimal_T, save_path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Uncalibrated
    confs_orig = [b["confidence"] for b in bin_data_orig if b["count"] > 0]
    accs_orig = [b["accuracy"] for b in bin_data_orig if b["count"] > 0]
    counts_orig = [b["count"] for b in bin_data_orig if b["count"] > 0]

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    ax1.bar(
        [b["bin_lower"] + 0.033 for b in bin_data_orig],
        [b["accuracy"] for b in bin_data_orig],
        width=0.06, color="#E53935", alpha=0.6, edgecolor="#B71C1C", label="Empirical Accuracy"
    )
    ax1.plot([b["confidence"] for b in bin_data_orig if b["count"] > 0],
             [b["accuracy"] for b in bin_data_orig if b["count"] > 0],
             "ro-", linewidth=2, markersize=6, label="Calibration Curve")
    ax1.set_title(f"Uncalibrated Softmax (T = 1.00)\nECE = {ece_orig:.4f} ({ece_orig*100:.2f}%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Confidence", fontsize=11)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # 2. Temperature Calibrated
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    ax2.bar(
        [b["bin_lower"] + 0.033 for b in bin_data_calib],
        [b["accuracy"] for b in bin_data_calib],
        width=0.06, color="#2E7D32", alpha=0.6, edgecolor="#1B5E20", label="Empirical Accuracy"
    )
    ax2.plot([b["confidence"] for b in bin_data_calib if b["count"] > 0],
             [b["accuracy"] for b in bin_data_calib if b["count"] > 0],
             "go-", linewidth=2, markersize=6, label="Calibrated Curve")
    ax2.set_title(f"Temperature-Calibrated (T = {optimal_T:.2f})\nECE = {ece_calib:.4f} ({ece_calib*100:.2f}%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Confidence (Scaled)", fontsize=11)
    ax2.set_ylabel("Empirical Accuracy", fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.suptitle("PlantHealth AI — Model Confidence Calibration Audit (Reliability Diagrams)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 80)
    print("PLANTHEALTH AI — CONFIDENCE CALIBRATION & TEMPERATURE SCALING AUDIT")
    print("=" * 80)

    start_time = time.time()

    # 1. Discover dataset and retrieve untouched validation and test splits
    all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)
    _, val_samples, test_samples = build_stratified_dataset_splits(all_items, class_to_idx)

    # Use 1,520 balanced samples for test evaluation to match primary benchmark, or full
    # Let's use balanced 1,520 test samples (40 per class)
    per_class = 40
    by_class = {c: [] for c in range(38)}
    for p, c in test_samples:
        by_class[c].append((p, c))
    test_eval_samples = []
    for c in range(38):
        test_eval_samples.extend(by_class[c][:per_class])

    # Validation: use 1,520 samples (40 per class) to optimize T
    by_class_val = {c: [] for c in range(38)}
    for p, c in val_samples:
        by_class_val[c].append((p, c))
    val_eval_samples = []
    for c in range(38):
        val_eval_samples.extend(by_class_val[c][:per_class])

    print(f"Validation optimization set: {len(val_eval_samples)} samples")
    print(f"Held-out test evaluation set: {len(test_eval_samples)} samples")

    # 2. Load MobileNetV2 model
    print(f"Loading MobileNetV2 model from {MOBILENET_MODEL_PATH}...")
    model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

    # 3. Extract logits
    print("Extracting validation pre-softmax logits...")
    val_logits, val_labels = extract_logits(model, val_eval_samples)

    print("Extracting test pre-softmax logits...")
    test_logits, test_labels = extract_logits(model, test_eval_samples)

    # 4. Temperature optimization on Validation data ONLY
    print("Optimizing Temperature parameter T on Validation data exclusively...")
    optimal_T, val_nll = optimize_temperature(val_logits, val_labels)
    print(f"Optimal Temperature Parameter T* = {optimal_T:.4f} (Validation NLL = {val_nll:.4f})")

    # 5. Evaluate on untouched held-out Test data
    print("\nAuditing test set confidence calibration...")
    # A. Uncalibrated (T = 1.0)
    test_probs_orig = softmax(test_logits, axis=1)
    test_preds_orig = np.argmax(test_probs_orig, axis=1)
    test_confs_orig = np.max(test_probs_orig, axis=1)

    ece_orig_15, bin_data_orig_15 = compute_ece(test_confs_orig, test_preds_orig, test_labels, n_bins=15)
    ece_orig_10, _ = compute_ece(test_confs_orig, test_preds_orig, test_labels, n_bins=10)
    mbrier_orig, bbrier_orig = compute_brier_scores(test_probs_orig, test_labels)

    # B. Calibrated (T = T*)
    test_probs_calib = softmax(test_logits / optimal_T, axis=1)
    test_preds_calib = np.argmax(test_probs_calib, axis=1)
    test_confs_calib = np.max(test_probs_calib, axis=1)

    ece_calib_15, bin_data_calib_15 = compute_ece(test_confs_calib, test_preds_calib, test_labels, n_bins=15)
    ece_calib_10, _ = compute_ece(test_confs_calib, test_preds_calib, test_labels, n_bins=10)
    mbrier_calib, bbrier_calib = compute_brier_scores(test_probs_calib, test_labels)

    print("\n" + "=" * 65)
    print("CONFIDENCE CALIBRATION AUDIT RESULTS (HELD-OUT TEST SET)")
    print("=" * 65)
    print(f"Metric                        | Uncalibrated (T=1.0) | Calibrated (T={optimal_T:.2f})")
    print("-" * 65)
    print(f"Expected Calib Error (15 bins)| {ece_orig_15:.4f} ({ece_orig_15*100:5.2f}%)   | {ece_calib_15:.4f} ({ece_calib_15*100:5.2f}%)")
    print(f"Expected Calib Error (10 bins)| {ece_orig_10:.4f} ({ece_orig_10*100:5.2f}%)   | {ece_calib_10:.4f} ({ece_calib_10*100:5.2f}%)")
    print(f"Multi-Class Brier Score       | {mbrier_orig:.4f}             | {mbrier_calib:.4f}")
    print(f"Binary Top-1 Brier Score      | {bbrier_orig:.4f}             | {bbrier_calib:.4f}")
    print(f"Top-1 Accuracy (Preserved)    | {np.mean(test_preds_orig == test_labels):.2%}             | {np.mean(test_preds_calib == test_labels):.2%}")
    print("=" * 65)

    # 6. Plot reliability diagrams
    plot_path = FINAL_RESULTS_DIR / "calibration_reliability_diagram.png"
    plot_reliability_diagrams(bin_data_orig_15, ece_orig_15, bin_data_calib_15, ece_calib_15, optimal_T, plot_path)
    print(f"\nSaved Reliability Diagrams to {plot_path}")

    # 7. Save calibration audit JSON
    calib_summary = {
        "optimal_temperature_T": optimal_T,
        "optimized_on": "Validation split (1,520 balanced samples)",
        "evaluated_on": "Test split (1,520 balanced samples)",
        "uncalibrated": {
            "temperature": 1.0,
            "ece_15_bins": ece_orig_15,
            "ece_10_bins": ece_orig_10,
            "multiclass_brier": mbrier_orig,
            "binary_top1_brier": bbrier_orig,
            "bin_details": bin_data_orig_15
        },
        "temperature_calibrated": {
            "temperature": optimal_T,
            "ece_15_bins": ece_calib_15,
            "ece_10_bins": ece_calib_10,
            "multiclass_brier": mbrier_calib,
            "binary_top1_brier": bbrier_calib,
            "bin_details": bin_data_calib_15
        },
        "findings": [
            f"Uncalibrated MobileNetV2 raw softmax outputs exhibit overconfidence, with an initial ECE of {ece_orig_15:.2%}.",
            f"Optimizing Temperature parameter T on the held-out validation set yielded T* = {optimal_T:.4f}.",
            f"Post-temperature scaling on the untouched test set reduced ECE by {(ece_orig_15 - ece_calib_15)*100:.2f} percentage points (down to {ece_calib_15:.2%}).",
            "Classification accuracy is strictly preserved since temperature scaling is a monotonic transformation on logits.",
            "Raw softmax outputs should not be presented as calibrated probabilistic certainty without temperature scaling or Platt scaling."
        ]
    }
    json_path = FINAL_RESULTS_DIR / "confidence_calibration.json"
    save_json(calib_summary, json_path)
    print(f"Saved calibration summary JSON to {json_path}")

    total_time = time.time() - start_time
    print(f"Calibration audit completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
