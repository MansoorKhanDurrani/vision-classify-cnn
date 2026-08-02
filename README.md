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
  - Built and trained baseline CNN (3 conv blocks + FC layers):
  - Trained for 3 epochs on the full training set (14,034 images)
  - Achieved 83.00% test accuracy - strong baseline before tuning
  - Model and loss curve saved (`models/baseline_cnn.pth`, `results/baseline_loss.png`)
  - Hyperparameter tuning (systematic, one variable at a time):
  - Best config: lr=0.001, epochs=6, dropout=0.3, filters=(16,32,64) - 84.23% test accuracy
  - lr=0.01 caused training to fail entirely (~random accuracy) - too high
  - Found and fixed overfitting with dropout (val loss was rising without it)
  - Tested smaller/larger architectures - default filter sizes remained best
- Augmentation on/off comparison:
  - With augmentation: train/val loss gap = 0.02, test acc 84.23%
  - Without augmentation: train/val loss gap = 0.32, test acc 83.00%
  - Confirmed augmentation's main benefit is reducing overfitting, not just accuracy
- Transfer learning (MobileNetV2, frozen feature extractor + fine-tuned classifier):
  - 87.77% test accuracy, val loss 0.334 - beats from-scratch CNN (84.23%)
  - Faster convergence (5 epochs vs 6), more stable loss curves
  - Proper evaluation (MobileNetV2, best model):
  - Overall test accuracy: 88% (3000 test images)
  - Confusion matrix generated - errors are semantically sensible, not random
  - Weakest class: mountain (75% recall) - confused with glacier 98 times (snowy/rocky terrain overlap)
  - Other notable confusions: street<->buildings (urban overlap), glacier->sea (blue/icy tone overlap)

## Currently working on

Day 3: building the Flask backend and simple frontend to serve predictions

- Day 2: hyperparameter tuning, transfer learning comparison (ResNet/MobileNet vs from-scratch CNN), and proper evaluation (confusion matrix, misclassification analysis)

- Proper evaluation (confusion matrix, misclassification analysis) on the best model (MobileNetV2 transfer    learning)

## Results Summary

| Model | Test Accuracy | Notes |
|---|---|---|
| From-scratch CNN (tuned) | 84.23% | lr=0.001, epochs=6, dropout=0.3 |
| **MobileNetV2 (transfer learning)** | **87.77%** | Best model - frozen features, fine-tuned classifier |

Full confusion matrix and per-class metrics in `results/confusion_matrix_mobilenet.png`.

## Future additions

- Hyperparameter tuning (architecture, learning rate, batch size, regularization)
- Transfer learning comparison (ResNet or MobileNet vs from-scratch CNN)
- Proper evaluation (confusion matrix, misclassification analysis)
- Web app (Flask backend + simple frontend) to serve predictions
- Web app (Flask backend + simple HTML/CSS/JS frontend) to serve predictions
## How to run

*(To be filled in once the training pipeline is complete)*