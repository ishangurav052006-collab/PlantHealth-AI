"""
PlantHealth AI - Phase 8: Live Prediction Verification Script
Verifies input preprocessing, class mapping, output probability distribution (shape 38, sum = 1.0),
and top-3 prediction accuracy on 20 unseen test images across distinct plant disease classes.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np

from config import DATA_DIR
from src.data_loader import build_stratified_dataset_splits, discover_dataset
from src.predict import PlantDiseasePredictor


def main():
    print("=" * 80)
    print("PLANTHEALTH AI - PHASE 8: LIVE INFERENCE VERIFICATION")
    print("=" * 80)

    # 1. Discover dataset and obtain untouched test split
    all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)
    _, _, test_samples = build_stratified_dataset_splits(all_items, class_to_idx)

    # 2. Select 20 distinct unseen test images across different classes
    selected = []
    seen_classes = set()
    for p, c in test_samples:
        if c not in seen_classes:
            selected.append((p, c))
            seen_classes.add(c)
        if len(selected) == 20:
            break

    print(f"Selected {len(selected)} unseen test images spanning {len(seen_classes)} unique classes.\n")

    # 3. Instantiate production predictor
    predictor = PlantDiseasePredictor(model_type="mobilenetv2")
    assert predictor.is_ready, "Predictor failed to initialize with trained weights!"

    correct_top1 = 0
    correct_top3 = 0

    print(f"{'#':<3} | {'True Class':<38} | {'Predicted Class':<38} | {'Conf':<7} | {'Top-1':<5} | {'Top-3':<5}")
    print("-" * 105)

    for i, (img_path, true_c) in enumerate(selected):
        res = predictor.predict(img_path, top_k=3)

        top1 = res["top_prediction"]
        top_k = res["top_k_predictions"]
        all_probs = res["all_probabilities"]
        prob_sum = float(sum(all_probs.values()))

        # Rigorous architectural assertions
        num_classes = len(all_probs)
        assert num_classes == 38, f"Output shape mismatch: expected 38 classes, got {num_classes}"
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum:.6f}"
        assert len(top_k) == 3, f"Expected top 3 predictions, got {len(top_k)}"
        assert top1["confidence"] > 0.0, "Prediction confidence must be positive"

        pred_idx = predictor.class_to_idx[top1["class_name"]]
        top3_indices = [predictor.class_to_idx[p["class_name"]] for p in top_k]

        is_top1 = (pred_idx == true_c)
        is_top3 = (true_c in top3_indices)

        if is_top1:
            correct_top1 += 1
        if is_top3:
            correct_top3 += 1

        true_name = idx_to_class[true_c]
        pred_name = top1["class_name"]
        conf_str = f"{top1['confidence'] * 100:.1f}%"
        t1_str = "YES" if is_top1 else "NO"
        t3_str = "YES" if is_top3 else "NO"

        print(f"{i+1:02d}  | {true_name:<38} | {pred_name:<38} | {conf_str:<7} | {t1_str:<5} | {t3_str:<5}")

    acc_top1 = correct_top1 / len(selected)
    acc_top3 = correct_top3 / len(selected)

    print("-" * 105)
    print(f"Top-1 Accuracy on 20 Unseen Images: {correct_top1}/20 ({acc_top1:.1%})")
    print(f"Top-3 Accuracy on 20 Unseen Images: {correct_top3}/20 ({acc_top3:.1%})")
    print(f"Output Dimension Check: PASS (38 classes)")
    print(f"Probability Normalization Check: PASS (sum = 1.000000)")
    print("=" * 80)


if __name__ == "__main__":
    main()
