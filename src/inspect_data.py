import os
from PIL import Image

train_dir = "data/seg_train/seg_train"
test_dir = "data/seg_test/seg_test"

print("=== Training set ===")
for class_name in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"{class_name}: {count} images")

print("\n=== Test set ===")
for class_name in sorted(os.listdir(test_dir)):
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        print(f"{class_name}: {count} images")

sample_class = sorted(os.listdir(train_dir))[0]
sample_dir = os.path.join(train_dir, sample_class)
sample_img_name = os.listdir(sample_dir)[0]
sample_img = Image.open(os.path.join(sample_dir, sample_img_name))

# Check if all images are the same size
sizes = set()
for class_name in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_path)[:50]:  # sample 50 per class for speed
        img = Image.open(os.path.join(class_path, img_name))
        sizes.add(img.size)

print(f"\nUnique sizes found (sampled): {sizes}")
print(f"\nSample image ({sample_img_name}) size: {sample_img.size}, mode: {sample_img.mode}")