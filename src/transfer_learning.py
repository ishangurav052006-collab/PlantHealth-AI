"""
PlantHealth AI - Transfer Learning Architecture
Implements MobileNetV2 (primary deployment model) and VGG16 architectures
with two-stage transfer learning: frozen base warmup followed by top-layer fine-tuning.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2, VGG16

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE_TRANSFER, LEARNING_RATE_FINE_TUNE


def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE_TRANSFER,
    dropout_rate: float = 0.4
) -> Tuple[keras.Model, keras.Model]:
    """
    Builds the primary MobileNetV2 transfer learning model.
    
    1. Loads ImageNet pre-trained weights without top classification layers.
    2. Freezes base weights for stage-1 classification head warmup.
    3. Adds GAP, BatchNorm, Dense, Dropout, and Softmax classification layers.
    
    Returns:
        Tuple[keras.Model, keras.Model]: (full_model, base_model)
    """
    # 1. Base Pre-trained Model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze all base layers initially

    # 2. Inputs & Classification Head
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Base feature extractor in inference mode
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dropout_rate, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantHealth_MobileNetV2")

    # 3. Compilation for initial training phase
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def setup_mobilenetv2_finetuning(
    model: keras.Model,
    base_model: keras.Model,
    fine_tune_at: int = 100,
    fine_tune_lr: float = LEARNING_RATE_FINE_TUNE
) -> keras.Model:
    """
    Unfreezes the upper layers of MobileNetV2 from index `fine_tune_at` upwards
    and re-compiles with a reduced learning rate for fine-tuning.
    """
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Keep BatchNormalization layers in inference mode to preserve ImageNet running statistics
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_vgg16_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE_TRANSFER,
    dropout_rate: float = 0.5
) -> Tuple[keras.Model, keras.Model]:
    """
    Builds the secondary VGG16 transfer learning model for experimental benchmark comparison.
    """
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="vgg_gap")(x)
    x = layers.Dense(512, activation="relu", name="vgg_dense_512")(x)
    x = layers.BatchNormalization(name="vgg_bn")(x)
    x = layers.Dropout(dropout_rate, name="vgg_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantHealth_VGG16")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def setup_vgg16_finetuning(
    model: keras.Model,
    base_model: keras.Model,
    unfreeze_blocks: int = 1,
    fine_tune_lr: float = LEARNING_RATE_FINE_TUNE
) -> keras.Model:
    """
    Unfreezes the top convolutional block (block5) of VGG16 for fine-tuning.
    """
    base_model.trainable = True
    
    # Freeze layers up to block5
    for layer in base_model.layers:
        if "block5" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
