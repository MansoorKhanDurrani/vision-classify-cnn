import torchvision.transforms as transforms

IMG_SIZE = 150

# Training transforms - includes augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms - NO augmentation, just resize + normalize
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    train_dataset = ImageFolder("data/seg_train/seg_train", transform=train_transforms)
    print("Classes:", train_dataset.classes)
    print("Number of training images:", len(train_dataset))

    # Peek at one transformed sample
    img, label = train_dataset[0]
    print("Transformed image shape:", img.shape)
    print("Transformed image dtype:", img.dtype)
    print("Label:", label, "->", train_dataset.classes[label])