import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from train import load_data

torch.manual_seed(42)
device = torch.device("cpu")

# --- Load pretrained MobileNetV2 ---
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Freeze the pretrained feature extractor (don't update these weights)
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final classifier layer for our 6 classes
model.classifier[1] = nn.Linear(model.last_channel, 6)
model = model.to(device)

# --- Data (reusing our existing pipeline) ---
train_loader, val_loader, test_loader = load_data(batch_size=32, use_augmentation=True)

criterion = nn.CrossEntropyLoss()
# Only optimize the new classifier layer's parameters
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"[transfer_mobilenet] Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

# --- Test evaluation ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"[transfer_mobilenet] Test Accuracy: {accuracy:.2f}%")

plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss - Transfer Learning (MobileNetV2)")
plt.legend()
plt.savefig("results/loss_transfer_mobilenet.png")
plt.close()

torch.save(model.state_dict(), "models/transfer_mobilenet.pth")
print("Model saved to models/transfer_mobilenet.pth")