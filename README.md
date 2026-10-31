# 🌿 PlantHealth AI – Plant Disease Classification using CNNs

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production-Ready](https://img.shields.io/badge/Status-Demonstrable-success.svg)]()

> **An enterprise-grade computer vision and deep learning platform engineered to diagnose crop foliar diseases across 38 classes, prevent data leakage in agricultural AI, and provide instant agronomic mitigation guidance.**

---

## 1. Project Introduction

**PlantHealth AI** is an end-to-end computer vision system designed for real-time plant disease classification. Farmers, agronomists, and researchers can upload photographs of crop foliage or select field samples to receive immediate disease classification, statistical prediction confidence, and pathogen-specific preventive measures.

---

## 2. Problem Statement

Crop diseases compromise between **20% and 40% of global agricultural yield**, threatening smallholder livelihoods and international food security. Traditional pathology diagnosis requires manual inspection by trained agronomists, culturing pathogens in laboratories, or expensive molecular PCR assays. These resources are rarely accessible in rural farming communities.

Automated image-based diagnosis through Deep Convolutional Neural Networks (CNNs) enables instant, non-invasive, scalable disease identification directly in the field.

---

## 3. Project Objectives

- **Full 38-Class Support:** Classify all 38 plant disease and healthy classes spanning 14 agricultural crops from the PlantVillage dataset.
- **Anti-Leakage Engineering:** Eliminate data leakage by grouping augmented leaf variants and near-duplicates by root sample UUIDs prior to stratified splitting.
- **Dual Architecture Strategy:**
  - Build and train a **Custom CNN from scratch** (4 convolutional blocks with Batch Normalization and Dropout).
  - Implement a **Two-Stage MobileNetV2 Transfer Learning** pipeline (ImageNet warmup followed by top-layer fine-tuning) as the primary deployment model.
- **Interactive AgTech Dashboard:** Provide a responsive Streamlit web application with a modern green and white color palette, confidence indicators, clinical descriptions, and agricultural prevention checklists.
- **Comprehensive Quality Assurance:** Maintain automated test suites, quantitative evaluation reports, confusion matrices, and reproducible Jupyter walkthroughs.

---

## 4. Supported Taxonomy & Classes (38 Total)

| Crop | Supported Conditions & Pathologies |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Citrus Greening (Huanglongbing) |
| **Peach** | Bacterial Spot, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 5. Technology Stack

- **Deep Learning Framework:** TensorFlow 2.x / Keras 3.x
- **Computer Vision & Processing:** OpenCV, Pillow, Scikit-learn
- **Data Engineering:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Testing:** Pytest
- **Runtime Environment:** Python 3.11 on Apple Silicon / CPU

---

## 6. Directory Layout

```
PlantHealth-AI/
├── app.py                     # Streamlit AgTech web application
├── config.py                  # Central configuration, paths, and disease taxonomy
├── requirements.txt           # Project dependencies
├── pytest.ini                 # Pytest configuration
├── PROJECT_AUDIT.md           # Formal repository audit report
├── README.md                  # Comprehensive documentation
│
├── src/                       # Core ML and pipeline modules
│   ├── __init__.py
│   ├── data_loader.py         # Discovery, anti-leakage splitting, and tf.data pipelines
│   ├── preprocessing.py      # Format validation, resizing, and normalization
│   ├── augmentation.py        # Runtime data augmentation (training split only)
│   ├── custom_cnn.py          # Custom CNN architecture from scratch
│   ├── transfer_learning.py   # MobileNetV2 and VGG16 transfer learning architectures
│   ├── train.py               # Unified training runner with callbacks
│   ├── evaluate.py            # Comprehensive evaluation suite
│   ├── predict.py             # Production inference engine
│   └── utils.py               # Visualizations, taxonomy parsing, and helpers
│
├── models/                    # Trained model weights & artifacts
│   ├── plantdiseasemodel.keras # Primary MobileNetV2 model
│   ├── custom_cnn.keras       # Scratch-trained Custom CNN
│   └── class_indices.json     # Deterministic class mapping
│
├── results/                   # Evaluation artifacts & plots
│   ├── evaluation_report.md   # Formal Markdown evaluation report
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── model_comparison.png
│   ├── prediction_samples.png
│   └── augmentation_samples.png
│
├── data/                      # PlantVillage dataset storage
│   ├── README.md              # Acquisition instructions
│   └── PlantVillage/          # 38 class directories
│
├── notebooks/                 # Executable Jupyter walkthroughs
│   ├── 01CustomCNN.ipynb
│   └── 02TransferLearning.ipynb
│
├── tests/                     # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_data_loader.py
│   ├── test_prediction.py
│   └── test_app.py
│
└── docs/                      # Architectural and technical documentation
    ├── MODEL_ARCHITECTURE.md
    ├── DATASET.md
    └── METHODOLOGY.md
```

---

## 7. Installation & Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
cd PlantHealth-AI

# Create isolated Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Dataset Acquisition & Discovery

The pipeline supports automatic discovery from local directory or archive:

```bash
# Option A: Download directly via Kaggle CLI
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p ~/Downloads/

# Option B: Extract automatically into data/PlantVillage
python -c "from src.data_loader import extract_dataset_archive; extract_dataset_archive()"
```

---

## 9. Model Training

### Train MobileNetV2 (Primary Model)
Executes two-stage transfer learning: frozen base warmup followed by top-layer fine-tuning:
```bash
python -m src.train --model mobilenetv2 --epochs 8 --max-per-class 70 --batch-size 32
```

### Train Custom CNN (From Scratch)
Trains the 4-block deep CNN with Batch Normalization and spatial Dropout:
```bash
python -m src.train --model custom_cnn --epochs 5 --max-per-class 40 --batch-size 32
```

### Train All Models Sequentially
```bash
python -m src.train --model all --epochs 8
```

---

## 10. Model Evaluation

Compute accuracy, precision, recall, macro/weighted F1, confusion matrices, and generate visualization artifacts:
```bash
python -m src.evaluate
```

Generated outputs in `results/`:
- `results/evaluation_report.md`
- `results/classification_report.csv`
- `results/confusion_matrix.png`
- `results/training_history.png`
- `results/model_comparison.png`
- `results/prediction_samples.png`

---

## 11. Command-Line Inference

Run instant foliar diagnosis on any image from the CLI:
```bash
python -m src.predict path/to/leaf_photo.jpg mobilenetv2
```

---

## 12. Running the Streamlit Web Application

Launch the AgTech dashboard:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Key Sections:
- **Home Dashboard:** Project overview, KPI cards, and end-to-end diagnostic workflow.
- **Disease Detection:** Upload custom leaf photos or test with preloaded agricultural specimens, inspect confidence ratings, and review preventive advice.
- **Model Performance:** Real-time benchmark tables, confusion matrix heatmaps, and training curves.
- **About & Architecture:** Architectural specifications, anti-leakage methodology, and pathogen etiology.

---

## 13. Running Automated Tests

Run the test suite with Pytest:
```bash
pytest -v
```

---

## 14. Project Limitations & Future Work

1. **Background Complexity in Real Fields:** PlantVillage images feature laboratory-controlled neutral backgrounds. Field deployment benefits from foreground leaf segmentation models (e.g., YOLO or SAM).
2. **Multi-Disease Co-Infection:** Currently framed as a single-label multi-class classification problem. Multi-label sigmoid heads will allow diagnosing co-occurring fungal and viral infections.
3. **Edge Optimization:** Converting `.keras` checkpoints into quantized INT8 TensorFlow Lite (`.tflite`) formats for on-device deployment on Android agricultural handhelds.
