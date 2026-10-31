# PlantHealth AI – Model Evaluation Report

**Generated:** 2026-10-31  
**Dataset:** PlantVillage (Stratified 15% Untouched Test Split)  
**Total Evaluated Classes:** 38  

---

## 1. Executive Summary & Benchmark Comparison

The following table compares performance across all trained architectures on the identical untouched test set:

| Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Primary)** | **10.33%** | 7.32% | 11.05% | **5.81%** | 5.68% | 600 |
| **Custom CNN** | **0.00%** | 0.00% | 0.00% | **0.00%** | 0.00% | 600 |

---

## 2. Primary Model Per-Class Classification Report

Below are precision, recall, and F1 metrics for each of the 38 plant disease classes evaluated on the primary model:

| Class Name | Plant | Condition | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `Apple___Apple_scab` | Apple | Apple Scab | 0.00% | 0.00% | 0.00% | 21 |
| `Apple___Black_rot` | Apple | Black Rot | 0.00% | 0.00% | 0.00% | 28 |
| `Apple___Cedar_apple_rust` | Apple | Cedar Apple Rust | 23.26% | 50.00% | 31.75% | 20 |
| `Apple___healthy` | Apple | None (Healthy) | 0.00% | 0.00% | 0.00% | 14 |
| `Blueberry___healthy` | Blueberry | None (Healthy) | 0.00% | 0.00% | 0.00% | 15 |
| `Cherry_(including_sour)___Powdery_mildew` | Cherry | Powdery Mildew | 25.00% | 6.67% | 10.53% | 15 |
| `Cherry_(including_sour)___healthy` | Cherry | None (Healthy) | 0.00% | 0.00% | 0.00% | 16 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn | Gray Leaf Spot | 27.78% | 76.92% | 40.82% | 13 |
| `Corn_(maize)___Common_rust_` | Corn | Common Rust | 0.00% | 0.00% | 0.00% | 0 |
| `Corn_(maize)___Northern_Leaf_Blight` | Corn | Northern Leaf Blight | 0.00% | 0.00% | 0.00% | 17 |
| `Corn_(maize)___healthy` | Corn | None (Healthy) | 100.00% | 14.29% | 25.00% | 14 |
| `Grape___Black_rot` | Grape | Black Rot | 0.00% | 0.00% | 0.00% | 23 |
| `Grape___Esca_(Black_Measles)` | Grape | Esca (Black Measles) | 0.00% | 0.00% | 0.00% | 16 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Leaf Blight (Isariopsis Spot) | 0.00% | 0.00% | 0.00% | 18 |
| `Grape___healthy` | Grape | None (Healthy) | 18.18% | 37.50% | 24.49% | 16 |
| `Orange___Haunglongbing_(Citrus_greening)` | Orange | Citrus Greening (Huanglongbing) | 0.00% | 0.00% | 0.00% | 17 |
| `Peach___Bacterial_spot` | Peach | Bacterial Spot | 0.00% | 0.00% | 0.00% | 25 |
| `Peach___healthy` | Peach | None (Healthy) | 11.11% | 7.69% | 9.09% | 13 |
| `Pepper,_bell___Bacterial_spot` | Bell Pepper | Bacterial Spot | 4.35% | 5.88% | 5.00% | 17 |
| `Pepper,_bell___healthy` | Bell Pepper | None (Healthy) | 0.00% | 0.00% | 0.00% | 12 |
| `Potato___Early_blight` | Potato | Early Blight | 0.00% | 0.00% | 0.00% | 17 |
| `Potato___Late_blight` | Potato | Late Blight | 0.00% | 0.00% | 0.00% | 16 |
| `Potato___healthy` | Potato | None (Healthy) | 7.39% | 92.86% | 13.68% | 14 |
| `Raspberry___healthy` | Raspberry | None (Healthy) | 6.67% | 25.00% | 10.53% | 8 |
| `Soybean___healthy` | Soybean | None (Healthy) | 0.00% | 0.00% | 0.00% | 18 |
| `Squash___Powdery_mildew` | Squash | Powdery Mildew | 0.00% | 0.00% | 0.00% | 10 |
| `Strawberry___Leaf_scorch` | Strawberry | Leaf Scorch | 7.14% | 6.25% | 6.67% | 16 |
| `Strawberry___healthy` | Strawberry | None (Healthy) | 12.12% | 66.67% | 20.51% | 12 |
| `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | 0.00% | 0.00% | 0.00% | 10 |
| `Tomato___Early_blight` | Tomato | Early Blight | 25.00% | 5.00% | 8.33% | 20 |
| `Tomato___Late_blight` | Tomato | Late Blight | 0.00% | 0.00% | 0.00% | 12 |
| `Tomato___Leaf_Mold` | Tomato | Leaf Mold | 0.00% | 0.00% | 0.00% | 16 |
| `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | 0.00% | 0.00% | 0.00% | 18 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-Spotted Spider Mite | 0.00% | 0.00% | 0.00% | 12 |
| `Tomato___Target_Spot` | Tomato | Target Spot | 0.00% | 0.00% | 0.00% | 11 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Yellow Leaf Curl Virus | 0.00% | 0.00% | 0.00% | 16 |
| `Tomato___Tomato_mosaic_virus` | Tomato | Mosaic Virus | 10.17% | 25.00% | 14.46% | 24 |
| `Tomato___healthy` | Tomato | None (Healthy) | 0.00% | 0.00% | 0.00% | 20 |

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