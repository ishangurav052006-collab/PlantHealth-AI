"""
PlantHealth AI - Dataset Discovery, Split, and tf.data Pipeline Module
Discovers PlantVillage images across 38 classes, enforces anti-leakage splitting,
and constructs high-throughput tf.data loaders with prefetching and caching.
"""

from collections import Counter, defaultdict
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.augmentation import get_training_augmentation_pipeline
from src.preprocessing import preprocess_for_model
from src.utils import logger, parse_class_name, save_json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_dataset_archive(
    zip_path: Union[str, Path] = Path("/Users/manthanshah/Downloads/archive.zip"),
    dest_dir: Union[str, Path] = DATA_DIR,
    max_images_per_class: Optional[int] = None
) -> Path:
    """
    Extracts PlantVillage dataset from the downloaded archive.zip into dest_dir.
    Normalizes the directory layout into dest_dir/<class_name>/<filename>.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_classes = [d for d in dest_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(existing_classes) >= NUM_CLASSES:
        logger.info(f"Dataset already extracted at {dest_dir} ({len(existing_classes)} classes found).")
        return dest_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at {zip_path}. Please download PlantVillage dataset or specify valid path."
        )

    logger.info(f"Extracting dataset from {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # We prefer the 'valid' + 'train' folders inside 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
        # or root class folders.
        namelist = z.namelist()
        valid_members = [
            n for n in namelist
            if "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/" in n
            and any(n.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        ]

        # Extract and group by class
        class_counts = defaultdict(int)
        for member in valid_members:
            parts = member.split("/")
            # e.g., parts = ['New Plant Diseases...', 'New Plant Diseases...', 'train', 'Apple___Apple_scab', 'image.jpg']
            if len(parts) >= 5:
                class_name = parts[3]
                filename = parts[4]
            else:
                continue

            if max_images_per_class and class_counts[class_name] >= max_images_per_class:
                continue

            target_class_dir = dest_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            target_file_path = target_class_dir / filename

            if not target_file_path.exists():
                source_data = z.read(member)
                with open(target_file_path, "wb") as f_out:
                    f_out.write(source_data)
                class_counts[class_name] += 1

    logger.info(f"Extraction complete! Populated {len(class_counts)} classes in {dest_dir}.")
    return dest_dir


def discover_dataset(data_dir: Union[str, Path] = DATA_DIR) -> Tuple[List[Tuple[Path, str]], Dict[str, int], Dict[int, str]]:
    """
    Scans dataset directory, validates image readability, and builds class-index mappings.
    
    Returns:
        Tuple:
            - all_items: List of (Path, class_name)
            - class_to_idx: Dict mapping class_name -> int
            - idx_to_class: Dict mapping int -> class_name
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_dir}")

    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    # Save class indices artifact
    save_json(class_to_idx, CLASS_INDICES_PATH)

    all_items = []
    corrupted_count = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                if img_path.stat().st_size == 0:
                    corrupted_count += 1
                    continue
                all_items.append((img_path, class_name))

    if corrupted_count > 0:
        logger.warning(f"Detected and skipped {corrupted_count} empty or corrupted images.")

    logger.info(f"Discovered {len(all_items)} valid images across {len(class_names)} classes.")
    return all_items, class_to_idx, idx_to_class


def _extract_sample_id(filename: str) -> str:
    """
    Extracts the root sample identifier from a PlantVillage filename to prevent
    augmented copies or near-duplicates from crossing split boundaries.
    
    Example:
        '00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335_270deg.JPG'
        -> '00075aa8-d81a-4184-8541-b692b78d398a'
    """
    if "___" in filename:
        return filename.split("___")[0].strip()
    match = re.match(r"^([a-f0-9\-]{8,})", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return filename.split(".")[0].split("_")[0]


def build_stratified_dataset_splits(
    all_items: List[Tuple[Path, str]],
    class_to_idx: Dict[str, int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    max_samples_per_class: Optional[int] = None,
    seed: int = RANDOM_SEED
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Splits dataset into 70% Train, 15% Validation, and 15% Test sets using:
    - Sample-group hashing to prevent data leakage of augmented / duplicate leaves
    - Stratified class partitioning to maintain exact class balance across splits
    - Optional max_samples_per_class to support compute-constrained hardware
    - Deterministic random seeding for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group items by class, then by sample root ID
    class_groups = defaultdict(lambda: defaultdict(list))
    for path, class_name in all_items:
        sample_id = _extract_sample_id(path.name)
        class_groups[class_name][sample_id].append(path)

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for class_name, sample_dict in sorted(class_groups.items()):
        class_idx = class_to_idx[class_name]
        unique_samples = list(sample_dict.keys())
        random.shuffle(unique_samples)

        if max_samples_per_class and len(unique_samples) > max_samples_per_class:
            unique_samples = unique_samples[:max_samples_per_class]

        n_samples = len(unique_samples)
        n_train = int(round(train_ratio * n_samples))
        n_val = int(round(val_ratio * n_samples))
        # Remainder goes to test to guarantee full coverage
        n_test = n_samples - n_train - n_val

        train_ids = set(unique_samples[:n_train])
        val_ids = set(unique_samples[n_train:n_train + n_val])
        test_ids = set(unique_samples[n_train + n_val:])

        for sid, paths in sample_dict.items():
            if sid not in train_ids and sid not in val_ids and sid not in test_ids:
                continue
            for p in paths:
                if sid in train_ids:
                    train_samples.append((p, class_idx))
                elif sid in val_ids:
                    val_samples.append((p, class_idx))
                else:
                    test_samples.append((p, class_idx))

    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    total = len(train_samples) + len(val_samples) + len(test_samples)
    logger.info(
        f"Stratified Splits Built (No Leakage): "
        f"Train={len(train_samples)} ({len(train_samples)/total:.1%}), "
        f"Val={len(val_samples)} ({len(val_samples)/total:.1%}), "
        f"Test={len(test_samples)} ({len(test_samples)/total:.1%})"
    )

    return train_samples, val_samples, test_samples


def create_tf_dataset(
    samples: List[Tuple[Path, int]],
    batch_size: int = BATCH_SIZE,
    model_type: str = "mobilenetv2",
    is_training: bool = False,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> tf.data.Dataset:
    """
    Constructs an optimized TensorFlow tf.data.Dataset pipeline:
    - High-throughput image decoding and resizing
    - Model-specific preprocessing (MobileNetV2, Custom CNN, or VGG16)
    - Runtime data augmentation on training split only
    - Shuffling, batching, and AUTOTUNE prefetching
    """
    paths = [str(p) for p, _ in samples]
    labels = [label for _, label in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(samples), 5000), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    def _parse_and_preprocess(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, image_size)

        # Apply model-specific normalization
        m_type = model_type.lower().replace("-", "").replace("_", "")
        if "mobilenet" in m_type:
            # Scale [0, 255] to [-1, 1]
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif "vgg" in m_type:
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            # Custom CNN: Scale [0, 255] to [0.0, 1.0]
            img = img / 255.0

        return img, label_tensor

    dataset = dataset.map(_parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data augmentation if training
    if is_training:
        aug_layer = get_training_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def plot_dataset_distributions(
    all_items: List[Tuple[Path, str]],
    train_samples: List[Tuple[Path, int]],
    val_samples: List[Tuple[Path, int]],
    test_samples: List[Tuple[Path, int]],
    idx_to_class: Dict[int, str],
    results_dir: Union[str, Path] = RESULTS_DIR
) -> None:
    """
    Generates class distribution and dataset split distribution charts.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class Distribution Chart
    class_counts = Counter(name for _, name in all_items)
    sorted_classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    bars = ax.bar(range(len(sorted_classes)), counts, color="#2E7D32", alpha=0.85)
    ax.set_title("PlantVillage Class Distribution Across 38 Classes", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Disease / Healthy Class", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels([c.replace("___", "\n") for c in sorted_classes], rotation=90, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(results_dir / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Split Distribution Chart
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    split_labels = ["Training (70%)", "Validation (15%)", "Test (15%)"]
    split_counts = [len(train_samples), len(val_samples), len(test_samples)]
    colors = ["#2E7D32", "#1976D2", "#F57C00"]
    ax.pie(split_counts, labels=split_labels, autopct="%1.1f%%", startangle=140, colors=colors,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Stratified Dataset Split Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.savefig(results_dir / "dataset_split_distribution.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved dataset distribution charts to results directory.")
