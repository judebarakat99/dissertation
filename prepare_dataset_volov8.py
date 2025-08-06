import os
import shutil
import random
from pathlib import Path

# Define dataset path
base_dir = Path("create_dataset/raw_images_RGB_yolov8")

# Paths to existing YOLOv8-format images/labels (already in train/)
image_dir = base_dir / "train" / "images"
label_dir = base_dir / "train" / "labels"

# Desired split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Collect image files
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

# Shuffle for randomness
random.shuffle(image_files)

# Split indices
total = len(image_files)
train_end = int(train_ratio * total)
val_end = train_end + int(val_ratio * total)

# Split datasets
splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# Create output folders
for split in splits:
    (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# Copy files with "skip if source is same as destination"
for split, files in splits.items():
    for img_path in files:
        label_path = label_dir / (img_path.stem + ".txt")

        dest_img = base_dir / split / "images" / img_path.name
        if img_path.resolve() != dest_img.resolve():
            shutil.copy(img_path, dest_img)

        dest_label = base_dir / split / "labels" / label_path.name
        if label_path.exists() and label_path.resolve() != dest_label.resolve():
            shutil.copy(label_path, dest_label)

print("✅ Dataset split complete.")
