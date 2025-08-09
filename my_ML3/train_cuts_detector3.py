#!/usr/bin/env python3
"""
Train Mask R-CNN on your COCO-format wound dataset.

Folder structure (as provided):
raw_images_RGB_annotated_coco/
├── train_split/               # training images
│   ├── *.jpg
│   └── train_split_annotations.json
├── val/                       # validation images
│   ├── *.jpg
│   └── val_annotations.json
└── test/                      # test images
    ├── *.jpg
    └── test_annotations.json
"""

import os
import time
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from PIL import Image
from pycocotools.coco import COCO


# -------------------- CONFIG --------------------
ROOT = "raw_images_RGB_annotated_coco"

TRAIN_DIR = os.path.join(ROOT, "train_split")
VAL_DIR   = os.path.join(ROOT, "val")
TEST_DIR  = os.path.join(ROOT, "test")

TRAIN_ANN = os.path.join(TRAIN_DIR, "train_split_annotations.json")
VAL_ANN   = os.path.join(VAL_DIR,   "val_annotations.json")
TEST_ANN  = os.path.join(TEST_DIR,  "test_annotations.json")

NUM_CLASSES   = 2          # background + wound
BATCH_SIZE    = 2
NUM_EPOCHS    = 10
LR            = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
STEP_SIZE     = 3
GAMMA         = 0.1

OUT_LATEST = "mask_cuts_detector.pth"
OUT_BEST   = "mask_cuts_detector_best.pth"

NUM_WORKERS = 2  # set to 0 if you hit dataloader issues
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ------------------------------------------------


# ---------------- COCO Dataset ------------------
class CutsCocoDataset(Dataset):
    """
    COCO-style dataset that returns:
      image (Tensor[C,H,W] in [0,1]),
      target dict with:
        - boxes  (FloatTensor[N,4])
        - labels (Int64Tensor[N])
        - masks  (UInt8Tensor[N,H,W])  values {0,1}
        - image_id (Int64Tensor[1])
    """
    def __init__(self, root: str, ann_file: str, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids  = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, info["file_name"])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks_list: List[torch.Tensor] = []

        for ann in anns:
            # bbox: x, y, w, h
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single foreground class
            masks_list.append(torch.as_tensor(self.coco.annToMask(ann), dtype=torch.uint8))

        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t  = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t  = torch.stack(masks_list, dim=0).to(torch.uint8)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def get_transforms(train: bool = False):
    t = [T.ToTensor()]
    if train:
        t += [T.RandomHorizontalFlip(0.5)]
    return T.Compose(t)


# --------------- Model builder ------------------
def build_maskrcnn(num_classes: int):
    # no pretrained heads (avoid internet), but you can change to weights="DEFAULT" for backbone pretraining if available
    model = maskrcnn_resnet50_fpn(weights=None)

    # box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# -------------- Dataloader Collate --------------
def collate_fn(batch):
    return tuple(zip(*batch))


# -------------- Train / Eval loops --------------
def train_one_epoch(model, optimizer, loader, device, epoch: int):
    model.train()
    start = time.time()
    running = 0.0

    for images, targets in loader:
        images  = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss.item())

    avg_loss = running / max(1, len(loader))
    print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  time={time.time()-start:.1f}s")
    return avg_loss


@torch.no_grad()
def eval_simple(model, loader, device):
    model.eval()
    total_imgs = 0
    total_dets = 0
    total_loss = 0.0

    for images, targets in loader:
        images = [im.to(device) for im in images]
        # forward for predictions
        outputs = model(images)
        total_imgs += len(outputs)
        total_dets += sum(len(o.get("boxes", [])) for o in outputs)

        # compute validation loss too (needs targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += float(sum(loss for loss in loss_dict.values()).item())

    avg_dets = total_dets / total_imgs if total_imgs > 0 else 0.0
    avg_loss = total_loss / max(1, len(loader))
    print(f"[Val] val_loss={avg_loss:.4f}  avg_dets/img={avg_dets:.2f}")
    return avg_loss, avg_dets


# -------------------- MAIN ----------------------
def main():
    os.makedirs(".", exist_ok=True)

    # datasets
    ds_train = CutsCocoDataset(TRAIN_DIR, TRAIN_ANN, transforms=get_transforms(train=True))
    ds_val   = CutsCocoDataset(VAL_DIR,   VAL_ANN,   transforms=get_transforms(train=False))
    ds_test  = CutsCocoDataset(TEST_DIR,  TEST_ANN,  transforms=get_transforms(train=False))

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val,   batch_size=2,            shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    dl_test  = DataLoader(ds_test,  batch_size=2,            shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # model
    model = build_maskrcnn(NUM_CLASSES).to(DEVICE)

    # resume if latest exists
    if os.path.exists(OUT_LATEST):
        print(f"Loading existing weights: {OUT_LATEST}")
        state = torch.load(OUT_LATEST, map_location=DEVICE)
        model.load_state_dict(state, strict=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, dl_train, DEVICE, epoch)
        val_loss, _ = eval_simple(model, dl_val, DEVICE)
        scheduler.step()

        # save latest
        torch.save(model.state_dict(), OUT_LATEST)
        print(f"Saved latest → {OUT_LATEST}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_BEST)
            print(f"✅ New best → {OUT_BEST}")

    # quick test pass
    print("\n--- Testing (forward only) ---")
    model.eval()
    with torch.no_grad():
        for images, _ in dl_test:
            images = [im.to(DEVICE) for im in images]
            _ = model(images)  # predictions computed; add metrics here if needed
            break
    print("Done.")

if __name__ == "__main__":
    print("Device:", DEVICE)
    main()
