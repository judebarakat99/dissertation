# train_cuts_detector.py
import os
import csv
import json
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from datetime import datetime
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks

# ------------------- Dataset -------------------
class CutsCocoDataset(Dataset):
    """
    COCO dataset (single class 'cut') with mask support.
    Expects polygons (COCO 'segmentation') or RLE in the annotations.
    """
    def __init__(self, root, annFile, transforms=None, filter_crowd=True):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.filter_crowd = filter_crowd

    def _ann_to_mask(self, ann, h, w):
        seg = ann.get("segmentation", None)
        if seg is None:
            return None
        # polygons (list) or RLE (dict)
        if isinstance(seg, list):
            rles = maskUtils.frPyObjects(seg, h, w)
            rle = maskUtils.merge(rles)
        else:
            rle = seg
        m = maskUtils.decode(rle)  # [H,W] uint8
        return m

    def __getitem__(self, index):
        img_id = self.ids[index]
        info = self.coco.loadImgs(img_id)[0]
        path = info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        W, H = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for ann in anns:
            if self.filter_crowd and ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single foreground class
            m = self._ann_to_mask(ann, H, W)
            if m is None:
                m = np.zeros((H, W), dtype=np.uint8)
            masks.append(m)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8) if len(masks) > 0 else torch.zeros((0, H, W), dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([img_id])}
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train):
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# ------------------- Model -------------------
def get_model(num_classes):
    # Handle different torchvision versions gracefully
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    except Exception:
        # older API
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace heads for our num_classes
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

# ------------------- Train -------------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(data_loader))

# ------------------- IoU helpers -------------------
def mask_iou_matrix(pred_masks_bool, gt_masks_bool):
    """
    pred_masks_bool: [Np, H, W] bool
    gt_masks_bool:   [Ng, H, W] bool
    returns IoU matrix [Np, Ng]
    """
    if pred_masks_bool.numel() == 0 or gt_masks_bool.numel() == 0:
        return torch.zeros((pred_masks_bool.shape[0], gt_masks_bool.shape[0]))
    Np, H, W = pred_masks_bool.shape
    Ng = gt_masks_bool.shape[0]
    pm = pred_masks_bool.reshape(Np, -1).float()
    gm = gt_masks_bool.reshape(Ng, -1).float()
    inter = pm @ gm.t()  # [Np, Ng]
    area_p = pm.sum(dim=1, keepdim=True)
    area_g = gm.sum(dim=1, keepdim=True).t()
    union = area_p + area_g - inter
    iou = inter / torch.clamp(union, min=1.0)
    return iou

def greedy_match(iou_mat, iou_thresh):
    """
    Greedy one-to-one matching by IoU.
    Returns matched_pred_idx, matched_gt_idx (lists).
    """
    matched_pred, matched_gt = [], []
    if iou_mat.numel() == 0:
        return matched_pred, matched_gt
    iou = iou_mat.clone()
    while True:
        max_val = iou.max()
        if max_val < iou_thresh:
            break
        idx = torch.nonzero(iou == max_val, as_tuple=False)[0]
        p, g = int(idx[0]), int(idx[1])
        matched_pred.append(p)
        matched_gt.append(g)
        iou[p, :] = -1
        iou[:, g] = -1
    return matched_pred, matched_gt

# ------------------- Metrics -------------------
@torch.no_grad()
def evaluate_metrics(model, data_loader, device, iou_type="mask", iou_thresh=0.5, score_working=0.5):
    """
    Computes:
      - Precision / Recall / F1 at score_working
      - AP@0.5 (IoU threshold)
    Uses greedy one-to-one matching per image.
    If iou_type='mask' but masks not available for an image, falls back to bbox IoU.
    """
    was_training = model.training
    model.eval()
    try:
        TP, FP, FN = 0, 0, 0
        all_scores, all_tp_flags = [], []
        total_gt_objects = 0

        for imgs, targets in data_loader:
            imgs_dev = [img.to(device) for img in imgs]
            outputs = model(imgs_dev)

            outs = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
            tars = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]

            for out, tar in zip(outs, tars):
                gt_boxes = tar["boxes"]
                gt_masks = tar["masks"].bool()  # [G,H,W]
                total_gt = gt_boxes.shape[0]
                total_gt_objects += total_gt

                pred_scores = out["scores"]
                pred_boxes  = out["boxes"]
                pred_masks  = out.get("masks", torch.zeros((0, 1, *gt_masks.shape[-2:]))).squeeze(1)

                # IoU matrix
                if iou_type == "mask" and gt_masks.numel() and pred_masks.numel():
                    iou_mat = mask_iou_matrix((pred_masks >= 0.5), gt_masks)
                else:
                    iou_mat = box_iou(pred_boxes, gt_boxes) if (pred_boxes.numel() and gt_boxes.numel()) else torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))

                # PR at working threshold
                keep = pred_scores >= score_working
                iou_work = iou_mat[keep] if iou_mat.numel() else torch.zeros((0, total_gt))
                matched_p, matched_g = greedy_match(iou_work, iou_thresh)
                tp = len(matched_p)
                fp = int(keep.sum().item()) - tp
                fn = total_gt - len(set(matched_g))
                TP += tp; FP += fp; FN += fn

                # AP@0.5 (global)
                order = torch.argsort(pred_scores, descending=True)
                used_gts = set()
                for idx in order.tolist():
                    all_scores.append(float(pred_scores[idx]))
                    ious_row = iou_mat[idx] if iou_mat.shape[0] > idx else torch.zeros((total_gt,))
                    best_gt, best_iou = -1, -1.0
                    for g in range(ious_row.numel()):
                        val = float(ious_row[g])
                        if val >= iou_thresh and g not in used_gts and val > best_iou:
                            best_iou, best_gt = val, g
                    if best_gt >= 0:
                        all_tp_flags.append(1)
                        used_gts.add(best_gt)
                    else:
                        all_tp_flags.append(0)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if len(all_scores) == 0 or total_gt_objects == 0:
            ap50 = 0.0
        else:
            order = np.argsort(-np.array(all_scores))
            tp = np.array(all_tp_flags)[order]
            fp = 1 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(1, total_gt_objects)
            precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)

            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([1.0], precisions, [0.0]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i-1] = max(mpre[i-1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        return {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1,
            "ap50": ap50,
            "total_gt_objects": total_gt_objects
        }
    finally:
        if was_training:
            model.train()

# ------------------- Visualization -------------------
def save_visualizations(img_tensors, outputs, out_dir="runs/vis", score_thresh=0.5, mask_thresh=0.5):
    os.makedirs(out_dir, exist_ok=True)
    for i, (img_t, out) in enumerate(zip(img_tensors, outputs)):
        img_cpu = (img_t.detach().cpu().clamp(0, 1))
        if len(out["scores"]) == 0:
            F.to_pil_image(img_cpu).save(os.path.join(out_dir, f"img_{i:04d}_no_dets.jpg"))
            continue
        scores = out["scores"].detach().cpu()
        keep_idx = scores >= score_thresh
        if keep_idx.sum() == 0:
            F.to_pil_image(img_cpu).save(os.path.join(out_dir, f"img_{i:04d}_no_confident_dets.jpg"))
            continue
        masks = out["masks"][keep_idx].detach().cpu().squeeze(1) >= mask_thresh  # [K,H,W] bool
        overlay = draw_segmentation_masks((img_cpu * 255).to(torch.uint8), masks=masks, alpha=0.5)
        F.to_pil_image(overlay).save(os.path.join(out_dir, f"img_{i:04d}_masks.jpg"))

# ------------------- Metrics logging -------------------
def ensure_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","split","TP","FP","FN","precision","recall","f1","ap50","total_gt_objects","timestamp"])

def write_metrics_row(csv_path, epoch, split, metrics):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, split,
            metrics["TP"], metrics["FP"], metrics["FN"],
            f"{metrics['precision']:.6f}", f"{metrics['recall']:.6f}", f"{metrics['f1']:.6f}",
            f"{metrics['ap50']:.6f}", metrics["total_gt_objects"],
            datetime.utcnow().isoformat() + "Z"
        ])

def save_metrics_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# ------------------- Main -------------------
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    train_dir = 'simulation_coco/train'
    val_dir   = 'simulation_coco/valid'
    test_dir  = 'simulation_coco/test'
    train_ann = os.path.join(train_dir, '_annotations.coco.json')
    val_ann   = os.path.join(val_dir,   '_annotations.coco.json')
    test_ann  = os.path.join(test_dir,  '_annotations.coco.json')

    dataset_train = CutsCocoDataset(train_dir, train_ann, transforms=get_transform(train=True))
    dataset_val   = CutsCocoDataset(val_dir,   val_ann,   transforms=get_transform(train=False))
    dataset_test  = CutsCocoDataset(test_dir,  test_ann,  transforms=get_transform(train=False))

    collate = lambda x: tuple(zip(*x))
    dl_train = DataLoader(dataset_train, batch_size=2, shuffle=True,  collate_fn=collate)
    dl_val   = DataLoader(dataset_val,   batch_size=2, shuffle=False, collate_fn=collate)
    dl_test  = DataLoader(dataset_test,  batch_size=2, shuffle=False, collate_fn=collate)

    model = get_model(num_classes=2)  # 1 class ('cut') + background
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    metrics_dir = "runs/metrics"
    vis_dir = "runs/vis"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "metrics_log.csv")
    ensure_csv(csv_path)

    num_epochs = 10
    best_val_ap50 = -1.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, dl_train, device)

        # Validate (mask IoU preferred; falls back to bbox when masks unavailable)
        val_metrics = evaluate_metrics(model, dl_val, device, iou_type="mask", iou_thresh=0.5, score_working=0.5)
        print(f"Epoch {epoch}/{num_epochs} | Train Loss {train_loss:.4f} | "
              f"Val P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
              f"F1 {val_metrics['f1']:.3f} AP50 {val_metrics['ap50']:.3f}")

        # Save JSON + CSV
        save_metrics_json(os.path.join(metrics_dir, f"val_metrics_epoch_{epoch:03d}.json"),
                          {"epoch": epoch, **val_metrics})
        write_metrics_row(csv_path, epoch, "val", val_metrics)

        # Save best by AP@0.5
        if val_metrics["ap50"] > best_val_ap50:
            best_val_ap50 = val_metrics["ap50"]
            torch.save(model.state_dict(), "cuts_maskrcnn_best.pth")
            print("Saved best model (by AP@0.5).")

    # --- Testing (load best), metrics + visualizations ---
    print("Loading best model for testing...")
    model.load_state_dict(torch.load("cuts_maskrcnn_best.pth", map_location=device))
    model.eval()

    test_metrics = evaluate_metrics(model, dl_test, device, iou_type="mask", iou_thresh=0.5, score_working=0.5)
    print(f"Test: P {test_metrics['precision']:.3f} R {test_metrics['recall']:.3f} "
          f"F1 {test_metrics['f1']:.3f} AP50 {test_metrics['ap50']:.3f}")
    save_metrics_json(os.path.join(metrics_dir, "test_metrics.json"), {"split": "test", **test_metrics})
    write_metrics_row(csv_path, "final", "test", test_metrics)

    # Save mask overlays for test images
    with torch.no_grad():
        for imgs, _ in dl_test:
            outs = model([im.to(device) for im in imgs])
            save_visualizations([im.cpu() for im in imgs], outs, out_dir=vis_dir, score_thresh=0.5, mask_thresh=0.5)

if __name__ == "__main__":
    main()
