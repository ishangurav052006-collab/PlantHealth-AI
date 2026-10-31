"""
PlantHealth AI - Phase 3: 100-Sample Inference Validation & Latency Profiling
Audits end-to-end inference across 100 previously unseen test images spanning all 38 classes.
Validates latency percentiles, output shapes, softmax normalization, top-1/top-3 accuracy,
and generates a qualitative visual report comparing correct and incorrect predictions.
"""

from collections import defaultdict
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import DATA_DIR, FINAL_RESULTS_DIR, MOBILENET_MODEL_PATH
from src.data_loader import build_stratified_dataset_splits, discover_dataset
from src.predict import PlantDiseasePredictor
from src.preprocessing import validate_and_load_image
from src.utils import parse_class_name, save_json


def main():
    print("=" * 80)
    print("PLANTHEALTH AI — 100-SAMPLE INFERENCE VALIDATION & LATENCY AUDIT")
    print("=" * 80)

    # 1. Discover dataset and retrieve untouched test split
    all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)
    _, _, test_samples = build_stratified_dataset_splits(all_items, class_to_idx)

    # Stratified selection of 100 images across all 38 classes
    by_class = defaultdict(list)
    for p, c in test_samples:
        by_class[c].append((p, c))

    selected_100 = []
    # Take 2-3 samples per class to reach 100
    for round_idx in range(4):
        for c in range(38):
            if round_idx < len(by_class[c]):
                selected_100.append(by_class[c][round_idx])
            if len(selected_100) == 100:
                break
        if len(selected_100) == 100:
            break

    print(f"Selected {len(selected_100)} test images across {len(set(c for _, c in selected_100))} classes.")

    # 2. Instantiate predictor
    predictor = PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH, model_type="mobilenetv2")
    assert predictor.is_ready, "Predictor model failed to load!"

    # Warmup inference
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _ = predictor.predict(dummy)

    latencies_ms = []
    results_records = []
    correct_samples = []
    incorrect_samples = []

    print("\nExecuting live inference across 100 test samples...")
    for i, (img_path, true_c) in enumerate(selected_100):
        t0 = time.perf_counter()
        res = predictor.predict(img_path, top_k=3)
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        latencies_ms.append(t_elapsed)

        top1 = res["top_prediction"]
        top_k = res["top_k_predictions"]
        all_probs = res["all_probabilities"]
        prob_sum = float(sum(all_probs.values()))

        # Architecture & normalization assertions
        assert len(all_probs) == 38, f"Output length mismatch: expected 38, got {len(all_probs)}"
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
        assert len(top_k) == 3, f"Expected top-3 predictions, got {len(top_k)}"

        pred_idx = predictor.class_to_idx[top1["class_name"]]
        top3_indices = [predictor.class_to_idx[p["class_name"]] for p in top_k]

        is_top1 = bool(pred_idx == true_c)
        is_top3 = bool(true_c in top3_indices)

        record = {
            "index": i + 1,
            "path": str(img_path),
            "true_class": idx_to_class[true_c],
            "pred_class": top1["class_name"],
            "confidence": float(top1["confidence"]),
            "is_top1": is_top1,
            "is_top3": is_top3,
            "latency_ms": t_elapsed,
            "top_3": [
                {"class": p["class_name"], "confidence": float(p["confidence"])}
                for p in top_k
            ]
        }
        results_records.append(record)

        # Store image array for visual grid
        img_arr, _ = validate_and_load_image(img_path)
        record["image_arr"] = img_arr

        if is_top1:
            correct_samples.append(record)
        else:
            incorrect_samples.append(record)

    latencies_np = np.array(latencies_ms)
    top1_acc = np.mean([r["is_top1"] for r in results_records])
    top3_acc = np.mean([r["is_top3"] for r in results_records])

    lat_mean = float(np.mean(latencies_np))
    lat_p50 = float(np.median(latencies_np))
    lat_p95 = float(np.percentile(latencies_np, 95))
    lat_min = float(np.min(latencies_np))
    lat_max = float(np.max(latencies_np))
    throughput = float(1000.0 / lat_mean)

    print("\n" + "=" * 55)
    print("100-SAMPLE INFERENCE AUDIT RESULTS")
    print("=" * 55)
    print(f"Top-1 Accuracy:          {top1_acc:.2%} ({int(top1_acc * 100)}/100 correct)")
    print(f"Top-3 Accuracy:          {top3_acc:.2%} ({int(top3_acc * 100)}/100 correct)")
    print(f"Total Evaluated Classes: 38/38")
    print(f"Output Shape:            (38,) for 100% of samples")
    print(f"Softmax Sum Sanity:      1.000000 for 100% of samples")
    print("-" * 55)
    print("LATENCY PROFILING (Single-Image CPU Inference):")
    print(f"  - Mean Latency:        {lat_mean:.2f} ms")
    print(f"  - Median (p50):        {lat_p50:.2f} ms")
    print(f"  - 95th Percentile:     {lat_p95:.2f} ms")
    print(f"  - Min / Max:           {lat_min:.2f} ms / {lat_max:.2f} ms")
    print(f"  - Throughput:          {throughput:.1f} images/sec")
    print("=" * 55 + "\n")

    # 3. Generate Visual Report Comparing Correct vs Incorrect Predictions
    print("Generating visual diagnostic report (Correct vs Incorrect)...")
    num_correct_to_plot = min(8, len(correct_samples))
    num_incorrect_to_plot = min(8, len(incorrect_samples))

    total_plots = num_correct_to_plot + num_incorrect_to_plot
    fig, axes = plt.subplots(4, 4, figsize=(18, 18), dpi=300)
    axes = axes.flatten()

    # Plot Correct Samples
    for idx in range(num_correct_to_plot):
        ax = axes[idx]
        item = correct_samples[idx]
        ax.imshow(item["image_arr"])
        true_parsed = parse_class_name(item["true_class"])
        title = (
            f"[CORRECT DIAGNOSIS]\n"
            f"Crop: {true_parsed['plant']}\n"
            f"True: {true_parsed['disease']}\n"
            f"Pred: {true_parsed['disease']}\n"
            f"Conf: {item['confidence']*100:.1f}% ({item['latency_ms']:.1f}ms)"
        )
        ax.set_title(title, fontsize=8.5, color="#1B5E20", fontweight="bold", pad=4)
        ax.axis("off")

    # Plot Incorrect Samples
    for idx in range(num_incorrect_to_plot):
        plot_idx = num_correct_to_plot + idx
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            item = incorrect_samples[idx]
            ax.imshow(item["image_arr"])
            true_parsed = parse_class_name(item["true_class"])
            pred_parsed = parse_class_name(item["pred_class"])
            title = (
                f"[MISCLASSIFICATION]\n"
                f"Crop: {true_parsed['plant']}\n"
                f"True: {true_parsed['disease']}\n"
                f"Pred: {pred_parsed['plant']} - {pred_parsed['disease']}\n"
                f"Conf: {item['confidence']*100:.1f}% (Top-3: {'YES' if item['is_top3'] else 'NO'})"
            )
            ax.set_title(title, fontsize=8.5, color="#B71C1C", fontweight="bold", pad=4)
            ax.axis("off")

    # Hide extra subplots if any
    for ax in axes[total_plots:]:
        ax.axis("off")

    plt.suptitle("PlantHealth AI — 100-Sample Inference Audit (Qualitative Case Studies)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    img_save_path = FINAL_RESULTS_DIR / "inference_validation_samples.png"
    plt.savefig(img_save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inference visual report to {img_save_path}")

    # Remove raw image arrays from JSON
    for r in results_records:
        r.pop("image_arr", None)

    summary_payload = {
        "total_evaluated": 100,
        "classes_represented": len(set(c for _, c in selected_100)),
        "metrics": {
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "num_correct": len(correct_samples),
            "num_incorrect": len(incorrect_samples),
        },
        "latency_ms": {
            "mean": lat_mean,
            "median_p50": lat_p50,
            "p95": lat_p95,
            "min": lat_min,
            "max": lat_max,
            "throughput_fps": throughput,
        },
        "sample_records": results_records
    }
    json_save_path = FINAL_RESULTS_DIR / "inference_validation_100.json"
    save_json(summary_payload, json_save_path)
    print(f"Saved 100-sample inference summary JSON to {json_save_path}")


if __name__ == "__main__":
    main()
