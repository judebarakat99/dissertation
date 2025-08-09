#!/usr/bin/env python3
import os
import csv
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image

# ---------------- Dataset (COCO with masks) ----------------
class CutsCocoDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, info["file_name"])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks_list = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single class: wound
            masks_list.append(torch.as_tensor(self.coco.annToMask(ann), dtype=torch.uint8))

        if len(boxes) == 0:
            w, h = img.size
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t  = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t  = torch.stack(masks_list, dim=0).to(torch.uint8)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# ---------------- Transforms ----------------
def get_transform(train):
    t = [T.ToTensor()]
    if train:
        t.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(t)

# ---------------- Model (Mask R-CNN) ----------------
def get_model(num_classes=2, score_thresh=0.5):
    # Tip: set weights="DEFAULT" if you want COCO-pretrained backbone (downloads once)
    model = maskrcnn_resnet50_fpn(weights=None)

    # replace box head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.roi_heads.score_thresh = score_thresh
    return model

# ---------------- Collate ----------------
def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------- One epoch train ----------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += float(losses.item())

    return total_loss / max(1, len(data_loader))

# ---------------- Simple validation ----------------
@torch.no_grad()
def evaluate(model, data_loader, device, compute_loss=True):
    # Detection count (eval path)
    model.eval()
    total_images, total_dets = 0, 0
    for imgs, _ in data_loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        total_images += len(outputs)
        total_dets += sum(len(o.get("boxes", [])) for o in outputs)
    avg_dets = total_dets / total_images if total_images > 0 else 0.0

    # Validation loss (switch to train() to get loss dict, but keep no_grad)
    val_loss = 0.0
    if compute_loss:
        was_training = model.training
        model.train()
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            val_loss += float(sum(loss for loss in loss_dict.values()).item())
        if not was_training:
            model.eval()
        val_loss = val_loss / max(1, len(data_loader))

    return val_loss, avg_dets

# ---------------- Main ----------------
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths (your structure)
    train_dir = 'raw_images_RGB_annotated_coco/train_split'
    val_dir   = 'raw_images_RGB_annotated_coco/val'
    test_dir  = 'raw_images_RGB_annotated_coco/test'

    train_ann = os.path.join(train_dir, 'train_split_annotations.json')
    val_ann   = os.path.join(val_dir,   'val_annotations.json')
    test_ann  = os.path.join(test_dir,  'test_annotations.json')

    # Datasets & loaders
    dataset_train = CutsCocoDataset(train_dir, train_ann, transforms=get_transform(train=True))
    dataset_val   = CutsCocoDataset(val_dir,   val_ann,   transforms=get_transform(train=False))
    dataset_test  = CutsCocoDataset(test_dir,  test_ann,  transforms=get_transform(train=False))

    data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True,
                                   num_workers=0, collate_fn=collate_fn)
    data_loader_val   = DataLoader(dataset_val,   batch_size=1, shuffle=False,
                                   num_workers=0, collate_fn=collate_fn)
    data_loader_test  = DataLoader(dataset_test,  batch_size=1, shuffle=False,
                                   num_workers=0, collate_fn=collate_fn)

    # Model
    model = get_model(num_classes=2, score_thresh=0.5).to(device)

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ---- Metrics CSV ----
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "training_metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "avg_dets_per_image"])

    # Train
    num_epochs = 10
    best_val = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device)
        val_loss, avg_dets = evaluate(model, data_loader_val, device, compute_loss=True)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Avg dets/img: {avg_dets:.2f}")

        # Append to CSV
        with open(metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{avg_dets:.4f}"])

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'mask_cuts_detector_best.pth')
            print("Saved best model → mask_cuts_detector_best.pth")

    # ---- quick test forward pass ----
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('mask_cuts_detector_best.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(data_loader_test):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)  # each output has 'masks' [N,1,H,W]
            print(f"Test image {i}: #instances={len(outputs[0].get('boxes', []))}")
            if i == 2:
                break

if __name__ == "__main__":
    main()
