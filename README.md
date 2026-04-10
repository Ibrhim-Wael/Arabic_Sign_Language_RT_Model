# 🤟 Arabic Sign Language Recognition — Real-Time CNN

A real-time Arabic Sign Language (ArSL) letter recognition system built from scratch using a custom CNN trained on two datasets, achieving **~98% test accuracy** across all 32 Arabic letter classes. Hand detection is handled by MediaPipe's HandLandmarker, and the top-3 predictions are displayed live on the webcam feed.

---

## Demo

> Real-time detection running in PyCharm — the model identifies hand gestures and overlays the predicted letter with confidence score.

![Demo Screenshot](assets/demo.png)

---

## How It Works

```
Webcam Frame
    │
    ▼
MediaPipe HandLandmarker
    │  detects hand landmarks → bounding box
    ▼
Crop & Preprocess (64×64, /255.0, BGR)
    │
    ▼
CNN Model (arabic_sign_model_v2.keras)
    │  predicts probabilities over 32 classes
    ▼
Temporal Smoothing (8-frame rolling average)
    │
    ▼
Display: top prediction + Top-3 sidebar
```

---

## Model Architecture

Built from scratch using TensorFlow / Keras — no pretrained backbone.

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | 32 units, Softmax |

- **Input:** 64 × 64 × 3 (BGR)
- **Output:** 32 classes (full Arabic alphabet)
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3)

---

## Training Pipeline

The model was trained in **two stages**:

### Stage 1 — Kaggle Dataset
Trained from scratch on the [Arabic Sign Language Dataset 2022](https://www.kaggle.com/datasets/your-link-here) (Kaggle).

- Images cropped from YOLO-format bounding boxes
- Resized to 64×64 and normalised to [0, 1]
- 70 / 15 / 15 train / val / test split
- Saved as `arabic_sign_model.h5`

### Stage 2 — Fine-tuning on a Second Dataset
The saved model was loaded and further trained on a second Arabic sign language dataset (unaugmented, 416px), covering 28 of the 32 classes.

- Class labels were remapped to match the original 32-class index scheme
- Same preprocessing pipeline
- Saved as `arabic_sign_model_v2.keras` ← **production model**

---

## Classes (32 Arabic Letters)

`ain` · `al` · `aleff` · `bb` · `dal` · `dha` · `dhad` · `fa` · `Qaaf` · `Ghain` · `Ha` · `Haa` · `Jeem` · `kaf` · `khaa` · `la` · `laam` · `meem` · `nun` · `ra` · `saad` · `seen` · `sheen` · `ta` · `taa` · `thaa` · `thal` · `toot` · `waw` · `ya` · `yaa` · `zay`

---

## Results

| Stage | Dataset | Accuracy |
|---|---|---|
| After Stage 1 | Kaggle test set | ~96% |
| After Stage 2 (fine-tuned) | Second dataset test set | **~98%** |

---

## Project Structure

```
arabic-sign-language/
├── notebook2cd44be66b.ipynb   # Stage 1: training on Kaggle dataset
├── Model.ipynb                # Stage 2: fine-tuning on second dataset
├── Rl.py                      # Real-time webcam detection
├── class_names.json           # 32 class labels (auto-generated)
├── hand_landmarker.task       # MediaPipe hand model (auto-downloaded)
├── assets/
│   └── demo.png
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/arabic-sign-language.git
cd arabic-sign-language
pip install -r requirements.txt
```

---

## Usage

1. Place your trained model at the path specified in `Rl.py` (or update `MODEL_PATH`).
2. Make sure `class_names.json` is at `CLASS_NAMES_PATH`.
3. Run:

```bash
python Rl.py
```

The MediaPipe hand model (~9MB) is downloaded automatically on first run.

**Controls:**
- `Q` — quit
- `C` — clear the prediction buffer

---

## Requirements

See [`requirements.txt`](requirements.txt)

---

## Datasets

- **Dataset 1:** [Arabic Sign Language Dataset 2022 — Kaggle](https://www.kaggle.com/datasets/your-link-here)
- **Dataset 2:** Secondary ArSL dataset (unaugmented, 416px images) — link unavailable

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-HandLandmarker-red)

---

## Author

**Ibrahim** — built as a personal deep learning project to bridge communication through technology.

> Feel free to open issues, suggest improvements, or fork the project!
