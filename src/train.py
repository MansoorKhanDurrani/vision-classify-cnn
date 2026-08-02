import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from model import SceneCNN
from preprocess import train_transforms, train_transforms_no_aug, eval_transforms


def load_data(batch_size=32, val_split=0.15, use_augmentation=True):
    train_tf = train_transforms if use_augmentation else train_transforms_no_aug

    # Two datasets pointing to the same folder, different transforms
    train_dataset_full = ImageFolder("data/seg_train/seg_train", transform=train_tf)
    val_dataset_full = ImageFolder("data/seg_train/seg_train", transform=eval_transforms)
    test_dataset = ImageFolder("data/seg_test/seg_test", transform=eval_transforms)

    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(train_dataset_full)), [train_size, val_size], generator=generator
    )

    train_subset = Subset(train_dataset_full, train_indices.indices)
    val_subset = Subset(val_dataset_full, val_indices.indices)  # uses eval_transforms - no augmentation

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(lr=0.001, epochs=3, dropout=0.0, batch_size=32, filters=(16, 32, 64), use_augmentation=True, run_name="baseline"):
    torch.manual_seed(42)
    device = torch.device("cpu")

    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, use_augmentation=use_augmentation)

    model = SceneCNN(num_classes=6, dropout=dropout, filters=filters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

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

        # Validation loss
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

        print(f"[{run_name}] Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Final test evaluation
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
    print(f"[{run_name}] Test Accuracy: {accuracy:.2f}%\n")

    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {run_name}")
    plt.legend()
    plt.savefig(f"results/loss_{run_name}.png")
    plt.close()

    torch.save(model.state_dict(), f"models/{run_name}.pth")

    return train_losses[-1], val_losses[-1], accuracy


if __name__ == "__main__":
    train_model(run_name="val_split_test")