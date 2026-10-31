"""
PlantHealth AI - Custom Convolutional Neural Network
Builds, compiles, and instantiates the from-scratch CNN architecture
optimized for 38-class plant disease classification.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE_CUSTOM


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE_CUSTOM,
    l2_reg: float = 1e-4
) -> keras.Model:
    """
    Constructs a deep, regularized convolutional network for plant disease diagnosis:
    - 4 Convolutional Feature Extraction stages with Batch Normalization
    - Spatial Max-Pooling and progressive Dropout for regularization
    - Global Average Pooling to reduce parameter count and eliminate spatial overfitting
    - Dense representation layer with Batch Normalization & Dropout
    - Softmax classification head over 38 classes
    """
    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1: 224x224 -> 112x112 (32 filters)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="drop1")(x)

    # Block 2: 112x112 -> 56x56 (64 filters)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3: 56x56 -> 28x28 (128 filters)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # Block 4: 28x28 -> 14x14 (256 filters)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizer, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, kernel_regularizer=regularizer, name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.40, name="drop_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PlantHealth_CustomCNN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
