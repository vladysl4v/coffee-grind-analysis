# Coffee Grind Fineness Estimation Pipeline

## Overview

This pipeline predicts coffee grind fineness (0–100) from images using:

1. Center crop (224×224)
2. Handcrafted texture features (selected subset)
3. CNN + MLP fusion model

The problem is fundamentally a **texture scale estimation problem**, not object detection.

---

# 1. Input Preprocessing

## Image crop

- Input: RGB image
- Operation:
  - Crop **center square**
  - Size: **224×224**

```python
crop = center_crop(image, size=224)
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
```

---

## Normalization

```python
gray = (gray - mean) / std
clip to [-3, 3]
rescale to [0, 255]
```

Purpose:
- Remove lighting variation
- Preserve texture structure

---

# 2. Selected Features (Final Set)

Total features: **19**

## 2.1 Morphological Features

Parameters:
- Structuring element: disk
- Radii: [13, 21]

Features:
- morph_closing_added_r21
- morph_closing_added_r13
- morph_opening_removed_r8

---

## 2.2 Lacunarity

Parameters:
- Box size: 10×10
- Threshold: median(gray)

Feature:
- lacunarity_box10

---

## 2.3 Autocorrelation Features

Features:
- autocorr_radial_mean
- autocorr_drop_010
- autocorr_drop_020
- autocorr_drop_050

---

## 2.4 GLCM Feature

Parameters:
- Levels: 32
- Distances: [1,2,4,8,16]
- Angles: [0°, 45°, 90°, 135°]

Feature:
- glcm_correlation_mean

---

## 2.5 FFT Features

Features:
- fft_low_power
- fft_mid_low_ratio
- fft_spectral_entropy

---

## 2.6 Local Entropy

Parameters:
- Radius: 5

Features:
- local_entropy_mean_r5
- local_entropy_p90_r5

---

## 2.7 Local Variance

Parameters:
- Window size: 9

Feature:
- local_std_p90_w9

---

## 2.8 Gradient Features

Features:
- grad_p99
- grad_std

---

## 2.9 Haar

Feature:
- haar_detail_mean

---

## 2.10 Laplace

Feature:
- laplace_mean_abs

---

## Feature Vector

- Shape: [batch_size, 19]

---

## Feature Standardization

```python
features = (features - mean_train) / std_train
```

---

# 3. Model Architecture

## Overview

Two-branch model:

Image → CNN → embedding  
Features → MLP → embedding  
Concatenate → regression head → fineness  

---

## 3.1 CNN Branch

Input:
- Shape: [B, 1, 224, 224]

Architecture:

```python
Conv2d(1, 32, 3, padding=1)
BatchNorm
ReLU
MaxPool(2)

Conv2d(32, 64, 3, padding=1)
BatchNorm
ReLU
MaxPool(2)

Conv2d(64, 128, 3, padding=1)
BatchNorm
ReLU
MaxPool(2)

Conv2d(128, 256, 3, padding=2, dilation=2)
BatchNorm
ReLU

AdaptiveAvgPool2d(1)
Flatten
```

Output:
- Shape: [B, 256]

---

## 3.2 Feature MLP Branch

Input:
- Shape: [B, 19]

Architecture:

```python
Linear(19 → 64)
BatchNorm
ReLU
Dropout(0.25)

Linear(64 → 32)
ReLU
```

Output:
- Shape: [B, 32]

---

## 3.3 Fusion + Regression

```python
Concat([256, 32]) → 288

Linear(288 → 128)
ReLU
Dropout(0.3)

Linear(128 → 32)
ReLU

Linear(32 → 1)
```

Output:
- Shape: [B, 1]

---

# 4. Training Setup

## Loss

```python
SmoothL1Loss()
```

Alternative:
```python
MSELoss()
```

---

## Optimizer

```python
AdamW(lr=1e-3, weight_decay=1e-4)
```

---

## Batch Size

- 16–32

---

## Regularization

- Dropout: 0.25–0.3
- Early stopping: recommended

---

# 5. Key Design Insights

This system relies on:

- Texture scale estimation

Important signals:

- gap size (morphology)
- spatial persistence (autocorrelation)
- clustering (lacunarity)
- frequency distribution (FFT)

---

# 6. Recommended Experiments

1. Features only (XGBoost / Ridge)
2. CNN only
3. CNN + features (final)

---

# 7. Expected Behavior

Fine grind:
- high entropy
- high gradients
- fast decorrelation

Coarse grind:
- large gaps
- strong autocorrelation
- low-frequency dominance

---

# 8. Notes for Integration

- Crop must be consistent (center-based)
- Feature normalization must use train statistics
- Feature order must remain fixed
- Model expects:
  - image tensor: float32, normalized
  - feature tensor: standardized
