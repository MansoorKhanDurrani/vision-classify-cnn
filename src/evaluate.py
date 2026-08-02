import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2
from sklearn.metrics import confusion_matrix, classification_report
from train import load_data

torch.manual_seed(42)
device = torch.device("cpu")

# --- Rebuild the same architecture and load saved weights ---
model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load("models/transfer_mobilenet.pth"))
model = model.to(device)
model.eval()

# --- Data ---
_, _, test_loader = load_data(batch_size=32, use_augmentation=True)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- Get all predictions ---
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- Plot confusion matrix ---
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - MobileNetV2 Transfer Learning")

for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("results/confusion_matrix_mobilenet.png")
plt.close()

print("\nConfusion matrix saved to results/confusion_matrix_mobilenet.png")

# --- Find most confused class pairs ---
print("\n=== Most common misclassifications ===")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    if cm_no_diag[idx] == 0:
        break
    print(f"Actual '{class_names[idx[0]]}' predicted as '{class_names[idx[1]]}': {cm_no_diag[idx]} times")
    cm_no_diag[idx] = 0