import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Paths
coco_json_path = "raw_images_RGB_annotated_coco/annotations/instances_default.json"
images_path = "raw_images_RGB_annotated_coco/images"
output_dir = Path("cnn_dataset")
output_dir.mkdir(exist_ok=True)

# Load COCO annotations
with open(coco_json_path) as f:
    coco_data = json.load(f)

# Create label folders
labels = ["wound"]  # Both "cuts" and "objects" are wounds
for split in ["train", "val", "test"]:
    for label in labels:
        (output_dir / split / label).mkdir(parents=True, exist_ok=True)

# Map image_id to filename
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Process annotations
crops = []
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, width, height]
    filename = image_id_to_filename[image_id]
    full_path = os.path.join(images_path, filename)

    image = cv2.imread(full_path)
    if image is None:
        continue

    x, y, w, h = list(map(int, bbox))
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        continue

    crops.append((crop, "wound"))

# Split data
train_set, test_set = train_test_split(crops, test_size=0.2, random_state=42)
val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42)

# Save crops
def save_crops(dataset, split):
    for i, (img, label) in enumerate(dataset):
        out_path = output_dir / split / label / f"{label}_{i}.jpg"
        cv2.imwrite(str(out_path), img)

save_crops(train_set, "train")
save_crops(val_set, "val")
save_crops(test_set, "test")

print("✅ Dataset prepared at ./cnn_dataset/")
