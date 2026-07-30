# Vision Classify CNN

A CNN-powered image classifier for natural scene recognition, wrapped in a simple web app.

## Goal

Classify natural scene images into 6 categories: **buildings, forest, glacier, mountain, sea, street** — using a CNN trained from scratch (and compared against a transfer learning model). The trained model will be served through a small web app where anyone can upload an image and get a prediction with a confidence score.

**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle)

## Done

- Set up repo structure (`data/`, `src/`, `notebooks/`, `results/`, `models/`, `app/`)
- Downloaded and extracted the Intel Image Classification dataset
- Inspected the dataset:
  - 6 classes, roughly balanced (~2200-2500 images/class train, ~440-550/class test) - no balancing needed
  - Image sizes are mostly 150x150, but not fully consistent (some 150x113, 150x110) - resize step required
  - RGB images
- Built preprocessing + augmentation pipeline:
  - Resize to 150x150, normalization (ImageNet stats)
  - Augmentation (training data only): horizontal flip, rotation, color jitter
  - Verified: 14,034 training images load correctly via `ImageFolder`, all 6 classes auto-detected

## Currently working on

- Building and training the first CNN (baseline model - Day 1 target is to get it running end to end)

## Future additions

- Hyperparameter tuning (architecture, learning rate, batch size, regularization)
- Transfer learning comparison (ResNet or MobileNet vs from-scratch CNN)
- Proper evaluation (confusion matrix, misclassification analysis)
- Web app (Flask backend + simple frontend) to serve predictions

## How to run

*(To be filled in once the training pipeline is complete)*
