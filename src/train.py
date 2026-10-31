"""
PlantHealth AI - Model Training Pipeline
Unified training runner supporting Custom CNN, MobileNetV2, and VGG16 with:
- ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks
- Two-stage Transfer Learning (Warmup -> Top-layer Fine-tuning)
- Non-leaking stratified validation
- Checkpoint persistence and history logging
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    CUSTOM_CNN_MODEL_PATH,
    DATA_DIR,
    DEFAULT_EPOCHS,
    LEARNING_RATE_CUSTOM,
    LEARNING_RATE_FINE_TUNE,
    LEARNING_RATE_TRANSFER,
    MOBILENET_MODEL_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    RESULTS_DIR,
    VGG16_MODEL_PATH,
)
from src.custom_cnn import build_custom_cnn
from src.data_loader import (
    build_stratified_dataset_splits,
    create_tf_dataset,
    discover_dataset,
    extract_dataset_archive,
    plot_dataset_distributions,
)
from src.transfer_learning import (
    build_mobilenetv2_model,
    build_vgg16_model,
    setup_mobilenetv2_finetuning,
    setup_vgg16_finetuning,
)
from src.utils import logger, plot_training_history, save_json


def get_training_callbacks(
    checkpoint_path: Path,
    monitor: str = "val_accuracy",
    patience_es: int = 5,
    patience_lr: int = 3
) -> List[keras.callbacks.Callback]:
    """Configures ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="max" if "acc" in monitor else "min",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            mode="max" if "acc" in monitor else "min",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode="max" if "acc" in monitor else "min",
            factor=0.2,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def train_custom_cnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = DEFAULT_EPOCHS,
    checkpoint_path: Path = CUSTOM_CNN_MODEL_PATH
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Trains the from-scratch Custom CNN."""
    logger.info("=" * 60)
    logger.info(f"STARTING CUSTOM CNN TRAINING (Target Epochs: {epochs})")
    logger.info("=" * 60)

    model = build_custom_cnn(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE_CUSTOM)
    callbacks = get_training_callbacks(checkpoint_path, monitor="val_accuracy")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(history_dict, RESULTS_DIR / "custom_cnn_history.json")
    plot_training_history(history_dict, RESULTS_DIR / "custom_cnn_history.png")

    logger.info(f"Custom CNN training completed. Best checkpoint saved to {checkpoint_path}")
    return model, history_dict


def train_mobilenetv2(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    warmup_epochs: int = 8,
    finetune_epochs: int = 7,
    checkpoint_path: Path = MOBILENET_MODEL_PATH
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """
    Two-stage transfer learning with MobileNetV2:
    - Stage 1: Frozen base warmup (train classification head)
    - Stage 2: Fine-tuning top convolutional layers with low learning rate
    """
    logger.info("=" * 60)
    logger.info(f"STARTING MOBILENETV2 TRAINING (Warmup: {warmup_epochs}, Fine-tune: {finetune_epochs})")
    logger.info("=" * 60)

    model, base_model = build_mobilenetv2_model(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE_TRANSFER)
    callbacks_warmup = get_training_callbacks(checkpoint_path, monitor="val_accuracy", patience_es=4)

    logger.info(">>> Stage 1: Training Classification Head (Pre-trained Base Frozen)...")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks_warmup,
        verbose=1
    )

    # Stage 2: Fine-Tuning
    if finetune_epochs > 0:
        logger.info(">>> Stage 2: Fine-Tuning MobileNetV2 Upper Layers (from layer 100)...")
        model = setup_mobilenetv2_finetuning(model, base_model, fine_tune_at=100, fine_tune_lr=LEARNING_RATE_FINE_TUNE)
        callbacks_finetune = get_training_callbacks(checkpoint_path, monitor="val_accuracy", patience_es=4)

        initial_epoch = len(h1.history["accuracy"])
        total_epochs = initial_epoch + finetune_epochs

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            callbacks=callbacks_finetune,
            verbose=1
        )

        combined_history = {}
        for k in h1.history:
            combined_history[k] = [float(v) for v in h1.history[k]] + [float(v) for v in h2.history.get(k, [])]
    else:
        combined_history = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    save_json(combined_history, RESULTS_DIR / "mobilenetv2_history.json")
    plot_training_history(combined_history, RESULTS_DIR / "training_history.png")

    logger.info(f"MobileNetV2 training completed. Best checkpoint saved to {checkpoint_path}")
    return model, combined_history


def train_vgg16(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    warmup_epochs: int = 5,
    finetune_epochs: int = 5,
    checkpoint_path: Path = VGG16_MODEL_PATH
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Trains VGG16 transfer learning model for experimental benchmark comparison."""
    logger.info("=" * 60)
    logger.info(f"STARTING VGG16 TRAINING (Warmup: {warmup_epochs}, Fine-tune: {finetune_epochs})")
    logger.info("=" * 60)

    model, base_model = build_vgg16_model(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE_TRANSFER)
    callbacks_warmup = get_training_callbacks(checkpoint_path, monitor="val_accuracy", patience_es=3)

    logger.info(">>> Stage 1: Training VGG16 Classification Head...")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks_warmup,
        verbose=1
    )

    if finetune_epochs > 0:
        logger.info(">>> Stage 2: Fine-Tuning VGG16 Block 5 Conv Layers...")
        model = setup_vgg16_finetuning(model, base_model, fine_tune_lr=LEARNING_RATE_FINE_TUNE)
        callbacks_finetune = get_training_callbacks(checkpoint_path, monitor="val_accuracy", patience_es=3)

        initial_epoch = len(h1.history["accuracy"])
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + finetune_epochs,
            callbacks=callbacks_finetune,
            verbose=1
        )
        combined = {k: [float(v) for v in h1.history[k]] + [float(v) for v in h2.history.get(k, [])] for k in h1.history}
    else:
        combined = {k: [float(v) for v in vals] for k, vals in h1.history.items()}

    save_json(combined, RESULTS_DIR / "vgg16_history.json")
    plot_training_history(combined, RESULTS_DIR / "vgg16_history.png")
    return model, combined


def main():
    parser = argparse.ArgumentParser(description="PlantHealth AI - Model Training Runner")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["custom_cnn", "mobilenetv2", "vgg16", "all"],
                        help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of total epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Dataset root directory")
    parser.add_argument("--archive-path", type=str, default="/Users/manthanshah/Downloads/archive.zip",
                        help="Path to PlantVillage archive.zip if extraction is required")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional cap on images per class for compute-constrained execution")
    args = parser.parse_args()

    # Step 1: Ensure dataset is extracted
    extract_dataset_archive(zip_path=args.archive_path, dest_dir=args.data_dir, max_images_per_class=args.max_per_class)

    # Step 2: Discover and split
    all_items, class_to_idx, idx_to_class = discover_dataset(args.data_dir)
    train_samples, val_samples, test_samples = build_stratified_dataset_splits(
        all_items, class_to_idx, max_samples_per_class=args.max_per_class
    )
    plot_dataset_distributions(all_items, train_samples, val_samples, test_samples, idx_to_class)

    # Step 3: Run selected training
    models_to_train = ["mobilenetv2", "custom_cnn"] if args.model == "all" else [args.model]

    for m_type in models_to_train:
        train_ds = create_tf_dataset(train_samples, batch_size=args.batch_size, model_type=m_type, is_training=True)
        val_ds = create_tf_dataset(val_samples, batch_size=args.batch_size, model_type=m_type, is_training=False)
        test_ds = create_tf_dataset(test_samples, batch_size=args.batch_size, model_type=m_type, is_training=False)

        if m_type == "custom_cnn":
            model, _ = train_custom_cnn(train_ds, val_ds, epochs=args.epochs)
        elif m_type == "mobilenetv2":
            warmup = max(1, args.epochs // 2)
            finetune = args.epochs - warmup
            model, _ = train_mobilenetv2(train_ds, val_ds, warmup_epochs=warmup, finetune_epochs=finetune)
        elif m_type == "vgg16":
            warmup = max(1, args.epochs // 2)
            finetune = args.epochs - warmup
            model, _ = train_vgg16(train_ds, val_ds, warmup_epochs=warmup, finetune_epochs=finetune)

        # Quick evaluation on untouched test set
        logger.info(f"Evaluating {m_type} on untouched test set ({len(test_samples)} samples)...")
        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
        logger.info(f">>> {m_type.upper()} FINAL TEST RESULTS: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4%}")


if __name__ == "__main__":
    main()
