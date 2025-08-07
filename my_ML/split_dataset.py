import os
import shutil
import random

# Paths
base_dir = 'raw_images_RGB_annotated_coco'
source_dir = os.path.join(base_dir, 'train')
train_dir = os.path.join(base_dir, 'train_split')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create destination folders if they don't exist
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# List all image files (assuming .jpg)
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle images
random.shuffle(image_files)

total = len(image_files)
train_count = int(0.7 * total)
val_count = int(0.15 * total)
test_count = total - train_count - val_count  # whatever remains

print(f'Total images: {total}')
print(f'Train: {train_count}, Validation: {val_count}, Test: {test_count}')

# Split the images
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Move files
def move_files(file_list, dest_dir):
    for file_name in file_list:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(dest_dir, file_name)
        shutil.move(src_path, dst_path)

move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Files moved successfully!")
