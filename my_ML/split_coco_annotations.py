import json
import os

# Paths
base_dir = 'raw_images_RGB_annotated_coco'
annotations_path = os.path.join(base_dir, 'train', '_annotations.coco.json')

train_dir = os.path.join(base_dir, 'train_split')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

def load_annotations(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_annotations(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def split_annotations(coco_data, image_filenames):
    # Create new COCO dict with filtered images and annotations
    filtered_images = [img for img in coco_data['images'] if img['file_name'] in image_filenames]
    image_ids = set(img['id'] for img in filtered_images)
    
    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
    
    # Copy all categories as is
    categories = coco_data['categories']
    
    return {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': categories,
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }

# Load original annotations
coco = load_annotations(annotations_path)

# Get image files in each split folder
train_images = set(os.listdir(train_dir))
val_images = set(os.listdir(val_dir))
test_images = set(os.listdir(test_dir))

# Split annotations
train_anns = split_annotations(coco, train_images)
val_anns = split_annotations(coco, val_images)
test_anns = split_annotations(coco, test_images)

# Save new annotation files
save_annotations(train_anns, os.path.join(train_dir, 'train_split_annotations.json'))
save_annotations(val_anns, os.path.join(val_dir, 'val_annotations.json'))
save_annotations(test_anns, os.path.join(test_dir, 'test_annotations.json'))

print("COCO annotation files have been split and saved.")
