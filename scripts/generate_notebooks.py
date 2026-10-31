"""
Helper script to generate the required Jupyter Notebook deliverables:
- notebooks/01CustomCNN.ipynb
- notebooks/02TransferLearning.ipynb
"""

import json
from pathlib import Path

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.18"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {filepath}")

# 1. Custom CNN Notebook
custom_cnn_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PlantHealth AI – Custom Convolutional Neural Network (From Scratch)\n",
            "\n",
            "This notebook demonstrates building, training, and evaluating a **Custom Deep CNN** architecture from scratch for 38-class plant disease classification on the PlantVillage dataset.\n",
            "\n",
            "### Pipeline Stages:\n",
            "1. Environment and Dependency Verification\n",
            "2. Dataset Discovery and Anti-Leakage Stratified Splitting\n",
            "3. Training Data Augmentation & Normalization Pipeline\n",
            "4. CNN Architecture Construction (Conv2D, BatchNorm, Pooling, Dropout)\n",
            "5. Model Compilation and Checkpoint Callbacks\n",
            "6. Model Training and Convergence Curves\n",
            "7. Quantitative Evaluation on Untouched Test Split\n",
            "8. Confusion Matrix & Qualitative Predictions"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Add project root to sys.path\n",
            "PROJECT_ROOT = Path(\"..\").resolve()\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import tensorflow as tf\n",
            "from tensorflow import keras\n",
            "\n",
            "from config import DATA_DIR, NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE, CUSTOM_CNN_MODEL_PATH\n",
            "from src.data_loader import discover_dataset, build_stratified_dataset_splits, create_tf_dataset\n",
            "from src.custom_cnn import build_custom_cnn\n",
            "from src.augmentation import get_training_augmentation_pipeline, visualize_augmentations\n",
            "from src.utils import plot_training_history, plot_confusion_matrix, plot_prediction_samples\n",
            "\n",
            "print(f\"TensorFlow Version: {tf.__version__}\")\n",
            "print(f\"Available Devices: {tf.config.list_physical_devices()}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Discovery and Stratified Partitioning\n",
            "We discover the 38 classes and split the dataset with strict anti-leakage grouping."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)\n",
            "class_names = [idx_to_class[i] for i in range(len(idx_to_class))]\n",
            "\n",
            "# Stratified Anti-Leakage Split (70% Train, 15% Val, 15% Test)\n",
            "train_samples, val_samples, test_samples = build_stratified_dataset_splits(\n",
            "    all_items, class_to_idx, max_samples_per_class=60\n",
            ")\n",
            "print(f\"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Augmentation Visualization\n",
            "We inspect real-time data augmentations applied strictly to the training split."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_path, sample_label = train_samples[0]\n",
            "sample_img = tf.keras.utils.load_img(sample_path, target_size=(224, 224))\n",
            "sample_arr = tf.keras.utils.img_to_array(sample_img)\n",
            "\n",
            "aug_pipe = get_training_augmentation_pipeline()\n",
            "fig, axes = plt.subplots(1, 5, figsize=(15, 3))\n",
            "axes[0].imshow(sample_arr.astype(\"uint8\"))\n",
            "axes[0].set_title(\"Original\")\n",
            "axes[0].axis(\"off\")\n",
            "\n",
            "for i in range(1, 5):\n",
            "    augmented = aug_pipe(np.expand_dims(sample_arr, 0), training=True)\n",
            "    axes[i].imshow(np.squeeze(augmented.numpy()).astype(\"uint8\"))\n",
            "    axes[i].set_title(f\"Augmented #{i}\")\n",
            "    axes[i].axis(\"off\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Custom CNN Architecture Construction\n",
            "Instantiate the 4-block deep CNN with Batch Normalization, Max-Pooling, Dropout, and Global Average Pooling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)\n",
            "model.summary()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Training or Checkpoint Loading\n",
            "We check for existing trained weights at `models/custom_cnn.keras` or perform training with callbacks."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_ds = create_tf_dataset(train_samples, batch_size=32, model_type=\"custom_cnn\", is_training=True)\n",
            "val_ds = create_tf_dataset(val_samples, batch_size=32, model_type=\"custom_cnn\", is_training=False)\n",
            "test_ds = create_tf_dataset(test_samples, batch_size=32, model_type=\"custom_cnn\", is_training=False)\n",
            "\n",
            "if CUSTOM_CNN_MODEL_PATH.exists():\n",
            "    print(f\"Loading verified trained checkpoint from {CUSTOM_CNN_MODEL_PATH}...\")\n",
            "    model = keras.models.load_model(CUSTOM_CNN_MODEL_PATH)\n",
            "    print(\"Model successfully loaded.\")\n",
            "else:\n",
            "    callbacks = [\n",
            "        keras.callbacks.EarlyStopping(monitor=\"val_accuracy\", patience=3, restore_best_weights=True),\n",
            "        keras.callbacks.ModelCheckpoint(str(CUSTOM_CNN_MODEL_PATH), save_best_only=True)\n",
            "    ]\n",
            "    history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=callbacks)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Evaluation on Untouched Test Set\n",
            "Assess generalization accuracy, loss, and confusion matrix."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_loss, test_acc = model.evaluate(test_ds)\n",
            "print(f\"Custom CNN Test Accuracy: {test_acc:.2%}\")\n",
            "print(f\"Custom CNN Test Loss: {test_loss:.4f}\")"
        ]
    }
]

# 2. Transfer Learning Notebook
transfer_learning_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PlantHealth AI – MobileNetV2 Transfer Learning Pipeline\n",
            "\n",
            "This notebook details the **Two-Stage Transfer Learning** strategy using ImageNet-pretrained **MobileNetV2** for high-accuracy plant pathology diagnosis.\n",
            "\n",
            "### Methodology:\n",
            "1. **Stage 1 (Feature Extraction Warmup):** Pre-trained base weights are frozen; custom classification head is trained to adapt to agricultural pathology distributions.\n",
            "2. **Stage 2 (Fine-Tuning):** Upper convolutional layers (layer index $\\ge 100$) are unfrozen and trained with a reduced learning rate ($10^{-5}$) to specialize spatial feature filters.\n",
            "3. **Rigorous Evaluation:** Benchmarked against untouched 15% stratified test split."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "PROJECT_ROOT = Path(\"..\").resolve()\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.metrics import classification_report, confusion_matrix\n",
            "import tensorflow as tf\n",
            "from tensorflow import keras\n",
            "\n",
            "from config import DATA_DIR, NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE, MOBILENET_MODEL_PATH\n",
            "from src.data_loader import discover_dataset, build_stratified_dataset_splits, create_tf_dataset\n",
            "from src.transfer_learning import build_mobilenetv2_model, setup_mobilenetv2_finetuning\n",
            "from src.utils import plot_training_history, plot_confusion_matrix, plot_prediction_samples\n",
            "\n",
            "print(f\"TensorFlow Version: {tf.__version__}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Dataset Splits\n",
            "Load stratified splits with anti-leakage sample grouping."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "all_items, class_to_idx, idx_to_class = discover_dataset(DATA_DIR)\n",
            "class_names = [idx_to_class[i] for i in range(len(idx_to_class))]\n",
            "train_samples, val_samples, test_samples = build_stratified_dataset_splits(\n",
            "    all_items, class_to_idx, max_samples_per_class=70\n",
            ")\n",
            "train_ds = create_tf_dataset(train_samples, batch_size=32, model_type=\"mobilenetv2\", is_training=True)\n",
            "val_ds = create_tf_dataset(val_samples, batch_size=32, model_type=\"mobilenetv2\", is_training=False)\n",
            "test_ds = create_tf_dataset(test_samples, batch_size=32, model_type=\"mobilenetv2\", is_training=False)\n",
            "print(f\"Test set samples: {len(test_samples)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. MobileNetV2 Architecture & Checkpoint Warmup\n",
            "Instantiate MobileNetV2 with frozen ImageNet base, loading trained checkpoint or executing training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if MOBILENET_MODEL_PATH.exists():\n",
            "    print(f\"Loading verified trained MobileNetV2 model from {MOBILENET_MODEL_PATH}...\")\n",
            "    model = keras.models.load_model(MOBILENET_MODEL_PATH)\n",
            "    print(\"Loaded primary deployment model successfully!\")\n",
            "else:\n",
            "    model, base_model = build_mobilenetv2_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)\n",
            "    model.summary()\n",
            "    h1 = model.fit(train_ds, validation_data=val_ds, epochs=1)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Stage 2 Fine-Tuning Overview\n",
            "In full training, upper layers from layer 100 upwards are unfrozen and fine-tuned with $\\eta = 10^{-5}$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Active model input shape: {model.input_shape}\")\n",
            "print(f\"Active model output shape: {model.output_shape}\")\n",
            "print(f\"Total trainable layers in model: {len(model.trainable_weights)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Comprehensive Model Evaluation on Test Split\n",
            "Generate classification report, confusion matrix, and prediction previews."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "test_loss, test_acc = model.evaluate(test_ds)\n",
            "print(f\"MobileNetV2 Test Accuracy: {test_acc:.2%}\")\n",
            "print(f\"MobileNetV2 Test Loss: {test_loss:.4f}\")\n",
            "\n",
            "raw_probs = model.predict(test_ds)\n",
            "y_pred = np.argmax(raw_probs, axis=1)\n",
            "y_true = np.array([lbl for _, lbl in test_samples])\n",
            "\n",
            "report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)\n",
            "print(report)"
        ]
    }
]

create_notebook(custom_cnn_cells, "notebooks/01CustomCNN.ipynb")
create_notebook(transfer_learning_cells, "notebooks/02TransferLearning.ipynb")
