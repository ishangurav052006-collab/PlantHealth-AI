# PlantHealth AI – Model Architecture Documentation

This document describes the architectural specifications, hyperparameter decisions, and structural design for the convolutional neural networks implemented in **PlantHealth AI**.

---

## 1. Primary Model: MobileNetV2 Transfer Learning

MobileNetV2 is an efficient, mobile-optimized convolutional neural network architecture built upon **Inverted Residual Blocks** and **Linear Bottlenecks**.

```
Input (224 × 224 × 3 RGB)
       │
MobileNetV2 Pretrained Base (ImageNet Weights, 154 layers)
       │
Global Average Pooling (GAP) -> 1280 features
       │
Batch Normalization (bn_head)
       │
Dense Layer (256 units, ReLU activation)
       │
Dropout Layer (Rate: 0.40)
       │
Softmax Classification Head (38 units, Categorical Probabilities)
```

### Two-Stage Transfer Learning Protocol
1. **Stage 1 – Head Warmup:**
   - Pretrained MobileNetV2 base weights are frozen (`trainable = False`).
   - The custom classification head is trained with learning rate $\eta = 10^{-4}$ using the Adam optimizer for 4-8 epochs.
   - Batch normalization statistics within the base model remain locked to ImageNet running means.
2. **Stage 2 – Top-Layer Fine-Tuning:**
   - MobileNetV2 layers from index $100$ upwards are unfrozen.
   - The entire network is fine-tuned with a delicate learning rate $\eta = 10^{-5}$.
   - All Batch Normalization layers throughout the network remain in inference mode to avoid destabilizing pre-trained moving statistics.

---

## 2. Custom CNN Architecture (Built From Scratch)

The Custom CNN evaluates learning capability from raw pixels without pre-existing ImageNet priors.

```
Layer (type)                     Output Shape          Param #     Details
================================================================================
input_image (InputLayer)        (None, 224, 224, 3)   0           RGB Leaf Image
--------------------------------------------------------------------------------
conv1_1 (Conv2D)                (None, 224, 224, 32)  896         3x3 conv, same, L2 reg
bn1_1 (BatchNormalization)      (None, 224, 224, 32)  128
relu1_1 (Activation)            (None, 224, 224, 32)  0           ReLU
conv1_2 (Conv2D)                (None, 224, 224, 32)  9,248       3x3 conv, same
bn1_2 (BatchNormalization)      (None, 224, 224, 32)  128
relu1_2 (Activation)            (None, 224, 224, 32)  0           ReLU
pool1 (MaxPooling2D)            (None, 112, 112, 32)  0           2x2 pool
drop1 (Dropout)                 (None, 112, 112, 32)  0           p = 0.20
--------------------------------------------------------------------------------
conv2_1 (Conv2D)                (None, 112, 112, 64)  18,496      3x3 conv, same
bn2_1 (BatchNormalization)      (None, 112, 112, 64)  256
relu2_1 (Activation)            (None, 112, 112, 64)  0
conv2_2 (Conv2D)                (None, 112, 112, 64)  36,928      3x3 conv, same
bn2_2 (BatchNormalization)      (None, 112, 112, 64)  256
relu2_2 (Activation)            (None, 112, 112, 64)  0
pool2 (MaxPooling2D)            (None, 56, 56, 64)    0           2x2 pool
drop2 (Dropout)                 (None, 56, 56, 64)    0           p = 0.25
--------------------------------------------------------------------------------
conv3_1 (Conv2D)                (None, 56, 56, 128)   73,856      3x3 conv, same
bn3_1 (BatchNormalization)      (None, 56, 56, 128)   512
relu3_1 (Activation)            (None, 56, 56, 128)   0
conv3_2 (Conv2D)                (None, 56, 56, 128)   147,584     3x3 conv, same
bn3_2 (BatchNormalization)      (None, 56, 56, 128)   512
relu3_2 (Activation)            (None, 56, 56, 128)   0
pool3 (MaxPooling2D)            (None, 28, 28, 128)   0           2x2 pool
drop3 (Dropout)                 (None, 28, 28, 128)   0           p = 0.30
--------------------------------------------------------------------------------
conv4_1 (Conv2D)                (None, 28, 28, 256)   295,168     3x3 conv, same
bn4_1 (BatchNormalization)      (None, 28, 28, 256)   1,024
relu4_1 (Activation)            (None, 28, 28, 256)   0
pool4 (MaxPooling2D)            (None, 14, 14, 256)   0           2x2 pool
drop4 (Dropout)                 (None, 14, 14, 256)   0           p = 0.35
--------------------------------------------------------------------------------
global_avg_pool (GAP)           (None, 256)           0           Spatial collapse
dense_features (Dense)          (None, 256)           65,792      Dense feature rep
bn_dense (BatchNormalization)   (None, 256)           1,024
relu_dense (Activation)         (None, 256)           0
drop_dense (Dropout)            (None, 256)           0           p = 0.40
classification_head (Dense)     (None, 38)            9,766       Softmax probabilities
================================================================================
```

---

## 3. Regularization & Optimization Strategy
1. **Weight Regularization:** $L_2$ kernel regularization ($\lambda = 10^{-4}$) on convolutional layers prevents kernel weights from exploding.
2. **Batch Normalization:** Stabilizes internal covariate shift across all convolutional stages, accelerating gradient flow.
3. **Dropout:** Progressive dropout rates ($0.20 \to 0.25 \to 0.30 \to 0.35 \to 0.40$) prevent co-adaptation of feature detectors.
4. **Adaptive Scheduling:** `ReduceLROnPlateau` decays learning rate by $5\times$ when validation performance plateaus for 3 epochs.
5. **Early Stopping:** Prevents overfitting by restoring best checkpoint weights.
