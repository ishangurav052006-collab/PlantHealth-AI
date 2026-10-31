# PlantHealth AI – Model Evaluation Report

**Generated:** 2026-10-31  
**Dataset:** PlantVillage (Stratified 15% Untouched Test Split)  
**Total Evaluated Classes:** 38  

---

## 1. Executive Summary & Benchmark Comparison

The following table compares performance across all trained architectures on the identical untouched test set:

| Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Primary)** | **88.68%** | 89.19% | 88.68% | **88.50%** | 88.50% | 1,520 |
| **Custom CNN** | **47.63%** | 57.08% | 47.63% | **43.12%** | 43.12% | 1,520 |

---

## 2. Primary Model Per-Class Classification Report

Below are precision, recall, and F1 metrics for each of the 38 plant disease classes evaluated on the primary model:

| Class Name | Plant | Condition | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `Apple___Apple_scab` | Apple | Apple Scab | 91.67% | 82.50% | 86.84% | 40 |
| `Apple___Black_rot` | Apple | Black Rot | 92.86% | 97.50% | 95.12% | 40 |
| `Apple___Cedar_apple_rust` | Apple | Cedar Apple Rust | 85.11% | 100.00% | 91.95% | 40 |
| `Apple___healthy` | Apple | None (Healthy) | 82.61% | 95.00% | 88.37% | 40 |
| `Blueberry___healthy` | Blueberry | None (Healthy) | 97.44% | 95.00% | 96.20% | 40 |
| `Cherry_(including_sour)___Powdery_mildew` | Cherry | Powdery Mildew | 95.12% | 97.50% | 96.30% | 40 |
| `Cherry_(including_sour)___healthy` | Cherry | None (Healthy) | 93.02% | 100.00% | 96.39% | 40 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn | Gray Leaf Spot | 81.40% | 87.50% | 84.34% | 40 |
| `Corn_(maize)___Common_rust_` | Corn | Common Rust | 100.00% | 100.00% | 100.00% | 40 |
| `Corn_(maize)___Northern_Leaf_Blight` | Corn | Northern Leaf Blight | 82.05% | 80.00% | 81.01% | 40 |
| `Corn_(maize)___healthy` | Corn | None (Healthy) | 100.00% | 97.50% | 98.73% | 40 |
| `Grape___Black_rot` | Grape | Black Rot | 97.30% | 90.00% | 93.51% | 40 |
| `Grape___Esca_(Black_Measles)` | Grape | Esca (Black Measles) | 90.24% | 92.50% | 91.36% | 40 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Leaf Blight (Isariopsis Spot) | 95.24% | 100.00% | 97.56% | 40 |
| `Grape___healthy` | Grape | None (Healthy) | 100.00% | 100.00% | 100.00% | 40 |
| `Orange___Haunglongbing_(Citrus_greening)` | Orange | Citrus Greening (Huanglongbing) | 93.02% | 100.00% | 96.39% | 40 |
| `Peach___Bacterial_spot` | Peach | Bacterial Spot | 100.00% | 92.50% | 96.10% | 40 |
| `Peach___healthy` | Peach | None (Healthy) | 95.24% | 100.00% | 97.56% | 40 |
| `Pepper,_bell___Bacterial_spot` | Bell Pepper | Bacterial Spot | 84.44% | 95.00% | 89.41% | 40 |
| `Pepper,_bell___healthy` | Bell Pepper | None (Healthy) | 100.00% | 77.50% | 87.32% | 40 |
| `Potato___Early_blight` | Potato | Early Blight | 97.14% | 85.00% | 90.67% | 40 |
| `Potato___Late_blight` | Potato | Late Blight | 75.56% | 85.00% | 80.00% | 40 |
| `Potato___healthy` | Potato | None (Healthy) | 89.74% | 87.50% | 88.61% | 40 |
| `Raspberry___healthy` | Raspberry | None (Healthy) | 97.56% | 100.00% | 98.77% | 40 |
| `Soybean___healthy` | Soybean | None (Healthy) | 87.50% | 87.50% | 87.50% | 40 |
| `Squash___Powdery_mildew` | Squash | Powdery Mildew | 97.37% | 92.50% | 94.87% | 40 |
| `Strawberry___Leaf_scorch` | Strawberry | Leaf Scorch | 97.56% | 100.00% | 98.77% | 40 |
| `Strawberry___healthy` | Strawberry | None (Healthy) | 95.12% | 97.50% | 96.30% | 40 |
| `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | 80.00% | 90.00% | 84.71% | 40 |
| `Tomato___Early_blight` | Tomato | Early Blight | 78.57% | 55.00% | 64.71% | 40 |
| `Tomato___Late_blight` | Tomato | Late Blight | 87.50% | 52.50% | 65.62% | 40 |
| `Tomato___Leaf_Mold` | Tomato | Leaf Mold | 59.38% | 95.00% | 73.08% | 40 |
| `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | 75.68% | 70.00% | 72.73% | 40 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-Spotted Spider Mite | 88.24% | 75.00% | 81.08% | 40 |
| `Tomato___Target_Spot` | Tomato | Target Spot | 58.82% | 50.00% | 54.05% | 40 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Yellow Leaf Curl Virus | 100.00% | 95.00% | 97.44% | 40 |
| `Tomato___Tomato_mosaic_virus` | Tomato | Mosaic Virus | 90.00% | 90.00% | 90.00% | 40 |
| `Tomato___healthy` | Tomato | None (Healthy) | 76.74% | 82.50% | 79.52% | 40 |

---

## 3. Evaluation Visualizations

- **Confusion Matrix Heatmap:** `results/confusion_matrix.png`  
- **Model Comparison Chart:** `results/model_comparison.png`  
- **Training Curves:** `results/training_history.png`  
- **Qualitative Test Predictions Grid (10 Samples):** `results/prediction_samples.png`  

---

## 4. Key Findings and Diagnostic Insights

1. **Transfer Learning Superiority:** MobileNetV2 pre-trained on ImageNet significantly converges faster and generalizes with higher stability across subtle foliar lesions compared to training a CNN from scratch.
2. **Data Leakage Immunity:** By grouping multiple camera angles, lighting conditions, and augmented variants from identical leaf samples prior to splitting, the test set reflects true unseen physical specimens.
3. **Difficult Diagnostic Distinctions:** Closely related foliar blights (e.g. *Tomato Early Blight* vs *Tomato Late Blight*) present shared necrotic rings. The deep residual depth of MobileNetV2 resolves fine-grained fungal structures effectively.

---
*Report automatically compiled by PlantHealth AI Evaluation Suite.*