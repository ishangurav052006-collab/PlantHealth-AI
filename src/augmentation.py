"""
PlantHealth AI - Augmentation Pipeline
Provides robust runtime data augmentation layers for training strictly.
Validation and test sets remain unaugmented.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import IMAGE_SIZE, RANDOM_SEED


def get_training_augmentation_pipeline() -> keras.Sequential:
    """
    Constructs a Keras Sequential preprocessing model applying:
    - Random rotation (up to 15 degrees / ~0.08 turn)
    - Random horizontal flipping
    - Random zoom (up to 10%)
    - Random translation (height & width by up to 10%)
    - Random brightness variation (+/- 15%)
    
    This layer operates purely during training (`training=True`)
    and behaves as an identity pass-through during inference (`training=False`).
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip(
            mode="horizontal",
            seed=RANDOM_SEED,
            name="aug_random_flip"
        ),
        layers.RandomRotation(
            factor=0.15,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=RANDOM_SEED,
            name="aug_random_rotation"
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            fill_mode="reflect",
            interpolation="bilinear",
            seed=RANDOM_SEED,
            name="aug_random_zoom"
        ),
        layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=RANDOM_SEED,
            name="aug_random_translation"
        ),
        layers.RandomBrightness(
            factor=0.15,
            value_range=(0, 255),
            seed=RANDOM_SEED,
            name="aug_random_brightness"
        )
    ], name="data_augmentation_pipeline")

    return data_augmentation


def visualize_augmentations(
    image: np.ndarray,
    save_path: Union[str, Path],
    num_variants: int = 5
) -> None:
    """
    Generates a figure displaying an original image alongside multiple
    augmented variations produced by the pipeline.
    """
    aug_pipeline = get_training_augmentation_pipeline()

    # Ensure shape is (1, H, W, C)
    if image.ndim == 3:
        img_batch = np.expand_dims(image, axis=0)
    else:
        img_batch = image

    # Generate variants
    fig, axes = plt.subplots(1, num_variants + 1, figsize=(18, 4), dpi=300)

    # Plot original
    axes[0].imshow(np.squeeze(img_batch).astype("uint8"))
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold", color="#1B5E20")
    axes[0].axis("off")

    for i in range(1, num_variants + 1):
        augmented = aug_pipeline(img_batch, training=True)
        aug_np = np.squeeze(augmented.numpy()).astype("uint8")
        axes[i].imshow(aug_np)
        axes[i].set_title(f"Augmented #{i}", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    plt.suptitle("Training Data Augmentation Variations (Rotation, Zoom, Flip, Shift, Brightness)",
                 fontsize=13, fontweight="bold", y=1.02)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
