"""
PlantHealth AI - Utility Module
Provides class name parsing, metric plotting, JSON persistence, and visualization helpers.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import CLASS_METADATA

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PlantHealthAI")


def parse_class_name(raw_name: str) -> Dict[str, str]:
    """
    Parses a PlantVillage directory or label name into structured components.
    
    Examples:
        - "Tomato___Early_blight" -> {"plant": "Tomato", "disease": "Early Blight", "status": "Diseased"}
        - "Pepper,_bell___healthy" -> {"plant": "Bell Pepper", "disease": "None (Healthy)", "status": "Healthy"}
        - "Corn_(maize)___Common_rust_" -> {"plant": "Corn", "disease": "Common Rust", "status": "Diseased"}
    """
    # Check if exact match exists in metadata
    if raw_name in CLASS_METADATA:
        meta = CLASS_METADATA[raw_name]
        return {
            "plant": meta["plant"],
            "disease": meta["disease"],
            "status": meta["status"],
            "raw_name": raw_name,
            "pathogen": meta.get("pathogen", "N/A"),
            "description": meta.get("description", ""),
            "prevention": meta.get("prevention", [])
        }

    # Fallback heuristic parser
    if "___" in raw_name:
        parts = raw_name.split("___")
        plant_raw, disease_raw = parts[0], parts[1]
    else:
        plant_raw, disease_raw = raw_name, "Unknown"

    # Clean plant name
    plant_clean = plant_raw.replace("_", " ").replace("(", "").replace(")", "").replace(",", "").strip()
    if "Pepper" in plant_clean and "bell" in plant_clean.lower():
        plant_clean = "Bell Pepper"
    elif "Corn" in plant_clean:
        plant_clean = "Corn"
    elif "Cherry" in plant_clean:
        plant_clean = "Cherry"

    # Clean disease name
    disease_clean = disease_raw.replace("_", " ").strip()
    is_healthy = "healthy" in disease_clean.lower()
    status = "Healthy" if is_healthy else "Diseased"
    if is_healthy:
        disease_clean = "None (Healthy)"

    return {
        "plant": plant_clean,
        "disease": disease_clean,
        "status": status,
        "raw_name": raw_name,
        "pathogen": "N/A",
        "description": "Plant disease classification sample.",
        "prevention": ["Consult an agricultural extension specialist for local management recommendations."]
    }


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save serializable data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON artifact to {path}")


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(history: Dict[str, List[float]], save_path: Union[str, Path]) -> None:
    """
    Plots training and validation accuracy and loss curves over epochs.
    """
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        logger.warning("Empty history passed to plot_training_history.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Accuracy subplot
    ax1.plot(epochs, history.get("accuracy", []), "o-", color="#2E7D32", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax1.plot(epochs, history.get("val_accuracy", []), "s--", color="#1565C0", label="Val Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy over Epochs", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Loss subplot
    ax2.plot(epochs, history.get("loss", []), "o-", color="#C62828", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax2.plot(epochs, history.get("val_loss", []), "s--", color="#EF6C00", label="Val Loss", linewidth=2)
    ax2.set_title("Cross-Entropy Loss over Epochs", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training history curves to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Union[str, Path]) -> None:
    """
    Plots a readable heatmap of the confusion matrix for 38 classes.
    """
    fig, ax = plt.subplots(figsize=(18, 16), dpi=300)
    sns.heatmap(
        cm,
        annot=False,  # 38 classes makes dense text clutter; colormap reveals off-diagonals cleanly
        cmap="Greens",
        fmt="d",
        xticklabels=[c.replace("___", " \n") for c in class_names],
        yticklabels=[c.replace("___", " \n") for c in class_names],
        cbar_kws={"label": "Sample Count"},
        ax=ax
    )
    ax.set_title("PlantVillage 38-Class Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix heatmap to {save_path}")


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], save_path: Union[str, Path]) -> None:
    """
    Plots comparative bar charts across models for Accuracy, Precision, Recall, and F1.
    """
    models = list(metrics_dict.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#2E7D32", "#1565C0", "#E65100", "#6A1B9A"]

    for i, model_name in enumerate(models):
        values = [metrics_dict[model_name].get(k, 0.0) for k in metric_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)], alpha=0.9)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2%}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparative Model Performance on Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model comparison plot to {save_path}")


def plot_prediction_samples(samples: List[Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """
    Plots a 2x5 grid of 10 test images showing:
    - Original image
    - Actual class
    - Predicted class
    - Prediction confidence
    - Correct / Incorrect badge
    """
    n = min(10, len(samples))
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 9), dpi=300)
    axes = axes.flatten()

    for idx in range(n):
        sample = samples[idx]
        ax = axes[idx]
        img = sample["image"]
        true_label = sample["true_label"]
        pred_label = sample["pred_label"]
        confidence = sample["confidence"]
        is_correct = sample["is_correct"]

        ax.imshow(img)
        ax.axis("off")

        badge = "CORRECT" if is_correct else "INCORRECT"
        badge_color = "#2E7D32" if is_correct else "#C62828"

        # Short clean names
        true_parsed = parse_class_name(true_label)
        pred_parsed = parse_class_name(pred_label)

        title_text = (
            f"True: {true_parsed['plant']} - {true_parsed['disease']}\n"
            f"Pred: {pred_parsed['plant']} - {pred_parsed['disease']}\n"
            f"Conf: {confidence:.1%} [{badge}]"
        )
        ax.set_title(title_text, fontsize=9, color=badge_color, fontweight="bold", pad=6)

    # Hide any unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sample prediction grid to {save_path}")
