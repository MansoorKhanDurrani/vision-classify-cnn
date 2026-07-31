import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import SceneCNN
from preprocess import train_transforms, eval_transforms

torch.manual_seed(42)

# --- Load data ---
train_dataset = ImageFolder("data/seg_train/seg_train", transform=train_transforms)
test_dataset = ImageFolder("data/seg_test/seg_test", transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# --- Model, loss, optimizer ---
device = torch.device("cpu")
model = SceneCNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training loop (small number of epochs, just to prove it runs end-to-end) ---
epochs = 3
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"=== Epoch {epoch+1} completed - Avg Loss: {epoch_loss:.4f} ===")

# --- Quick test accuracy check ---
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

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# --- Save loss plot ---
plt.plot(train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Avg Training Loss")
plt.title("Baseline CNN - Training Loss")
plt.savefig("results/baseline_loss.png")
plt.close()

# --- Save model ---
torch.save(model.state_dict(), "models/baseline_cnn.pth")
print("Model saved to models/baseline_cnn.pth")