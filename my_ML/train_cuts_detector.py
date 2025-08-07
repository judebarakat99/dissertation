import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image

# --- Dataset class for COCO formatted dataset ---
class CutsCocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # one class only

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# --- Simple transforms ---
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

# --- Load model and modify head ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Training one epoch ---
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for imgs, targets in data_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

# --- Validation ---
def evaluate(model, data_loader, device):
    model.eval()
    total_images = 0
    total_detections = 0

    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)

            total_images += len(imgs)
            total_detections += sum([len(o['boxes']) for o in outputs])

    avg_detections = total_detections / total_images if total_images > 0 else 0
    print(f"Evaluation: {total_images} images, {total_detections} total detections, avg {avg_detections:.2f} per image")
    return avg_detections  # or return 0 if you still want to save best model based on some metric


# --- Testing / inference ---
def test_model(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, _ in data_loader:  # targets may be dummy or empty during testing
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)
            results.extend(outputs)
    return results

# --- Main function ---
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    train_dir = 'raw_images_RGB_annotated_coco/train_split'
    val_dir = 'raw_images_RGB_annotated_coco/val'
    test_dir = 'raw_images_RGB_annotated_coco/test' 

    train_ann = os.path.join(train_dir, 'train_split_annotations.json')
    val_ann = os.path.join(val_dir, 'val_annotations.json')
    test_ann = os.path.join(test_dir, 'test_annotations.json') 


    dataset_train = CutsCocoDataset(train_dir, train_ann, transforms=get_transform(train=True))
    dataset_val = CutsCocoDataset(val_dir, val_ann, transforms=get_transform(train=False))
    dataset_test = CutsCocoDataset(test_dir, test_ann, transforms=get_transform(train=False))

    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=2)  # 1 class + background
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 10
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device)
        val_loss = evaluate(model, data_loader_val, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cuts_detector_best.pth')
            print("Saved best model.")

    # --- Testing ---
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('cuts_detector_best.pth'))
    results = test_model(model, data_loader_test, device)

    # Example: print boxes, labels, and scores for each test image
    for i, output in enumerate(results):
        print(f"Test Image {i}:")
        print("Boxes:", output['boxes'])
        print("Labels:", output['labels'])
        print("Scores:", output['scores'])
        print("------")

if __name__ == "__main__":
    main()




