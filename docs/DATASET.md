# PlantVillage Dataset Documentation

## 1. Overview
The **PlantVillage** dataset is a globally recognized agricultural computer vision benchmark comprising **87,867 leaf images** representing **14 crop species** across **38 distinct classes** (encompassing fungal, bacterial, viral infections, mite damage, and healthy foliage).

---

## 2. Supported Crops and Disease Taxonomy

| Crop | Classes Supported | Pathogen Etiology |
| :--- | :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy | *Venturia inaequalis*, *Botryosphaeria obtusa*, *Gymnosporangium juniperi-virginianae* |
| **Blueberry** | Healthy | N/A |
| **Cherry** | Powdery Mildew, Healthy | *Podosphaera clandestina* |
| **Corn (Maize)** | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | *Cercospora zeae-maydis*, *Puccinia sorghi*, *Exserohilum turcicum* |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy | *Guignardia bidwellii*, *Phaeomoniella chlamydospora*, *Pseudocercospora vitis* |
| **Orange** | Citrus Greening (Huanglongbing) | *Candidatus Liberibacter asiaticus* |
| **Peach** | Bacterial Spot, Healthy | *Xanthomonas arboricola pv. pruni* |
| **Bell Pepper** | Bacterial Spot, Healthy | *Xanthomonas euvesicatoria* |
| **Potato** | Early Blight, Late Blight, Healthy | *Alternaria solani*, *Phytophthora infestans* |
| **Raspberry** | Healthy | N/A |
| **Soybean** | Healthy | N/A |
| **Squash** | Powdery Mildew | *Podosphaera xanthii* |
| **Strawberry** | Leaf Scorch, Healthy | *Diplocarpon earlianum* |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | *Xanthomonas*, *Alternaria linariae*, *Phytophthora infestans*, *Passalora fulva*, *Septoria lycopersici*, *Tetranychus urticae*, *Begomovirus*, *Tobamovirus* |

---

## 3. Data Leakage Prevention Strategy
A significant vulnerability in automated plant pathology pipelines stems from naive train/val/test random splitting on augmented datasets. In datasets where multiple rotations, zoom variations, and lighting permutations exist of identical leaves, naive random splitting allows augmented variants of test leaves to leak into the training partition.

### Mitigation Architecture
1. **Root Sample ID Extraction:** PlantVillage filenames contain root sample UUID hashes (e.g. `00075aa8-d81a-4184-8541-b692b78d398a`).
2. **Group Hashing:** All physical leaf variations sharing a root UUID are assigned as an indivisible atomic unit.
3. **Stratified Splitting:** Partitioning occurs across the 38 classes preserving exact 70% Training, 15% Validation, and 15% Test proportions.
4. **Leakage Verification:** Automated unit tests mathematically verify that `train_ids ∩ test_ids = ∅` and `train_ids ∩ val_ids = ∅`.

---

## 4. Preprocessing & Augmentation
- **Standardized Input:** Images are resized using Lanczos interpolation to $224 \times 224 \times 3$ RGB.
- **Normalization:**
  - MobileNetV2: Scaled to $[-1, 1]$ via `tf.keras.applications.mobilenet_v2.preprocess_input`.
  - Custom CNN: Rescaled to $[0, 1]$ via pixel division by 255.0.
  - VGG16: Zero-centered with ImageNet mean channels.
- **Dynamic Training Augmentation:** Applied strictly on the training stream (`training=True`):
  - Random Rotation ($\pm 15^\circ$)
  - Horizontal Flipping
  - Random Zoom ($\pm 10\%$)
  - Random Translation ($\pm 10\%$)
  - Random Brightness ($\pm 15\%$)
