# Vision Classify

A natural scene image classifier powered by a CNN, fine-tuned via transfer learning, and served through an interactive web app. Upload any photo and get an instant prediction across 6 categories: **buildings, forest, glacier, mountain, sea, and street**.

![Status](https://img.shields.io/badge/status-complete-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red)

---

## Overview

This project trains a convolutional neural network to classify natural scene photographs, then wraps it in a Flask web app so anyone can try it. It compares a CNN built from scratch against a fine-tuned pretrained model, and evaluates both honestly rather than just reporting the best number.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| From-scratch CNN (tuned) | 84.23% | 3 conv blocks, lr=0.001, epochs=6, dropout=0.3 |
| **MobileNetV2 (transfer learning)** | **87.77%** | Frozen pretrained features + fine-tuned classifier head — **best model, used in the live app** |

**Confusion matrix (best model):** 88% overall accuracy across 3,000 test images. Misclassifications cluster around semantically related classes rather than being random:

- `mountain` → `glacier` (98 cases) — both feature snowy/rocky terrain
- `street` → `buildings` (74 cases) — both are urban scenes
- `glacier` → `sea` (29 cases) — both have blue/icy tones

Full confusion matrix: `results/confusion_matrix_mobilenet.png`

## What I Tried and Learned

**Hyperparameter tuning** (one variable at a time):
- lr=0.01 caused training to fail entirely (~18% accuracy, near-random) — too high
- lr=0.0001 was too slow to converge in a reasonable number of epochs
- lr=0.001 was the sweet spot
- Without dropout, the model overfit after ~3-4 epochs (val loss rising while train loss kept falling); dropout=0.3 fixed this
- Smaller (8,16,32) and larger (32,64,128) filter sizes were both tried — neither beat the original (16,32,64) architecture

**Augmentation on vs. off** (same config, only this changed):
- With augmentation: train/val loss gap = 0.02 → strong generalization, 84.23% accuracy
- Without augmentation: train/val loss gap = 0.32 → clear overfitting, 83.00% accuracy
- Takeaway: augmentation's biggest win here was reducing overfitting, not raw accuracy

**Transfer learning vs. from-scratch:**
- MobileNetV2 (ImageNet-pretrained, feature extractor frozen, only classifier head fine-tuned) beat the from-scratch CNN by ~3.5%, converged faster (5 vs 6 epochs), and generalized better (val loss below train loss)
- Pretraining already encodes general visual features (edges, textures, shapes) that transfer directly to natural scene recognition

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision, NumPy, Matplotlib, Scikit-learn
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Version Control:** Git, GitHub — feature-branch workflow (feature branches → `dev` → `main`)

## Project Structure

- `app/app.py` — Flask backend, loads the model once at startup and serves predictions
- `app/templates/index.html` — frontend page
- `app/static/` — CSS, JS, and one sample image per class
- `data/` — dataset (gitignored, see setup below)
- `src/inspect_data.py` — dataset inspection (class balance, image sizes)
- `src/preprocess.py` — transforms: augmentation + normalization
- `src/model.py` — from-scratch CNN architecture
- `src/train.py` — reusable training function
- `src/run_experiments.py` — hyperparameter tuning experiments
- `src/train_transfer.py` — MobileNetV2 fine-tuning
- `src/evaluate.py` — confusion matrix + classification report
- `models/` — saved model weights (`.pth`)
- `results/` — loss curves, confusion matrix

## How to Run

**1. Clone the repo**

git clone https://github.com/MansoorKhanDurrani/vision-classify-cnn.git
cd vision-classify-cnn


**2. Install dependencies**

pip install -r requirements.txt


**3. Get the dataset**

Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle, extract it into `data/`, so you end up with:

data/seg_train/seg_train/<classes>/
data/seg_test/seg_test/<classes>/


**4. (Optional) Retrain the model**

python src/train_transfer.py

This saves the trained model to `models/transfer_mobilenet.pth`, which the app loads.

**5. Run the web app**

python app/app.py

Open `http://127.0.0.1:5000` in your browser.

## Development Roadmap

- [x] Project Initialization
- [x] Dataset Collection
- [x] Data Preprocessing
- [x] Data Augmentation
- [x] CNN Architecture Development
- [x] Model Training
- [x] Hyperparameter Tuning
- [x] Model Evaluation
- [x] Flask Backend Development
- [x] Frontend Development
- [x] Final Deployment (local)

## Future Additions

- Deploy publicly (e.g. Hugging Face Spaces) for a shareable live link
- Grad-CAM heatmaps to visualize what the model is focusing on
- Handle out-of-scope images ("not confident this is any of my 6 classes")
- Top-3 predictions with confidence bars, not just the top pick

## Author

**Mansoor Khan**
BS Software Engineering Student
Learning Machine Learning and Deep Learning through practical projects.