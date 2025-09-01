#!/usr/bin/env python3
# run_detection_masks.py
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ----------------------------
# Utilities
# ----------------------------
def find_test_images(sim_coco_dir: Path) -> List[Path]:
    cand_dirs = [
        sim_coco_dir / "test" / "images",  # common COCO export
        sim_coco_dir / "test"              # flat folder
    ]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    images = []
    for d in cand_dirs:
        if d.is_dir():
            for ext in exts:
                images.extend(sorted(d.glob(ext)))
    return images

def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def pick_font(size: int = 16) -> Optional[ImageFont.FreeTypeFont]:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
        return None

# ----------------------------
# Model loading
# ----------------------------
def _try_import_training_model() -> Optional[object]:
    """
    Try to import a model-constructor from your training script (best way
    to ensure the architecture matches your weights).
    """
    here = Path(__file__).parent.resolve()
    sys.path.insert(0, str(here))
    try:
        import train_cuts_detector as tcd
    except Exception:
        return None

    for name in ("get_model","create_model","build_model","make_model","get_detector","create_detector"):
        fn = getattr(tcd, name, None)
        if callable(fn):
            return fn
    return None

def build_fallback_maskrcnn(num_classes: int = 2):
    # Fallback: standard Mask R-CNN backbone
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        # Replace heads to fit our classes
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model
    except Exception:
        # Older torchvision fallback
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model

def load_detector(weights_path: Path, device: torch.device):
    """
    Load model weights and return (model, class_names).
    """
    ckpt = torch.load(str(weights_path), map_location="cpu")

    # Try to get class names from checkpoint metadata
    class_names = None
    if isinstance(ckpt, dict):
        for key in ("classes","class_names","names","labels"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                class_names = list(ckpt[key])
                break

    # Decide #classes (include background if missing)
    if class_names is None:
        num_classes = 2  # ["__background__", "cut"]
        class_names = ["__background__", "cut"]
    else:
        if class_names and class_names[0].lower() in ("__background__","background","bg"):
            num_classes = len(class_names)
        else:
            num_classes = len(class_names) + 1
            class_names = ["__background__"] + class_names

    # Prefer training constructor for exact architecture match
    ctor = _try_import_training_model()
    if ctor:
        try:
            model = ctor(num_classes=num_classes)
        except TypeError:
            model = ctor()  # if ctor ignores num_classes, still okay
    else:
        model = build_fallback_maskrcnn(num_classes=num_classes)

    # Load state dict
    state = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt
    # Strip possible "module." prefixes
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    model.eval().to(device)
    return model, class_names

# ----------------------------
# Inference
# ----------------------------
@torch.inference_mode()
def run_inference_on_image(
    model,
    image_path: Path,
    device: torch.device,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5,
    mask_thresh: float = 0.5
) -> Tuple[Image.Image, dict]:
    """
    Returns:
      - original PIL image
      - dict with filtered tensors (cpu): boxes [N,4], scores [N], labels [N], masks [N,H,W](bool)
    """
    img_pil = Image.open(str(image_path)).convert("RGB")
    img_tensor = F.to_tensor(img_pil).to(device)

    out = model([img_tensor])[0]  # torchvision detection API

    boxes  = out.get("boxes",  torch.empty((0,4), device=device))
    scores = out.get("scores", torch.empty((0,), device=device))
    labels = out.get("labels", torch.empty((0,), dtype=torch.long, device=device))
    masks  = out.get("masks",  torch.empty((0,1,*img_tensor.shape[-2:]), device=device))

    # Confidence filter
    keep = scores >= conf_thresh
    boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]

    # NMS on boxes
    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes, scores, labels, masks = boxes[keep_idx], scores[keep_idx], labels[keep_idx], masks[keep_idx]

    # Threshold masks to bool [N,H,W]
    if masks.numel() > 0:
        masks = (masks.squeeze(1) >= mask_thresh)
    else:
        masks = torch.zeros((0, img_tensor.shape[-2], img_tensor.shape[-1]), dtype=torch.bool, device=device)

    return img_pil, {
        "boxes": boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "labels": labels.detach().cpu(),
        "masks": masks.detach().cpu()
    }

def _random_color_from_id(idx: int) -> Tuple[int,int,int]:
    # Stable pseudo-random color from an integer id
    h = hash(idx)
    return ((h >> 0) & 255, (h >> 8) & 255, (h >> 16) & 255)

def overlay_masks_and_boxes(
    img_pil: Image.Image,
    preds: dict,
    class_names: List[str],
    draw_boxes: bool = False
) -> Image.Image:
    """
    Overlays boolean masks with translucency. Optionally draws boxes + labels.
    """
    img_t = F.to_tensor(img_pil).clamp(0,1)  # [3,H,W] float in [0,1]
    masks = preds["masks"]  # [N,H,W] bool
    if masks.shape[0] > 0:
        # If you want custom colors per instance:
        colors = [_random_color_from_id(i) for i in range(masks.shape[0])]
        # torchvision expects either a single color or a list per mask (RGB 0-255)
        overlay = draw_segmentation_masks((img_t*255).to(torch.uint8), masks=masks, alpha=0.5, colors=colors)
    else:
        overlay = (img_t*255).to(torch.uint8)

    vis = F.to_pil_image(overlay)

    if draw_boxes and preds["boxes"].shape[0] > 0:
        draw = ImageDraw.Draw(vis)
        font = pick_font(16)
        for i, (box, score, label) in enumerate(zip(preds["boxes"], preds["scores"], preds["labels"])):
            x1, y1, x2, y2 = map(float, box.tolist())
            color = _random_color_from_id(i)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            lid = int(label)
            cls = class_names[lid] if 0 <= lid < len(class_names) else f"id_{lid}"
            text = f"{cls} {float(score):.2f}"
            if font:
                tw, th = draw.textbbox((0,0), text, font=font)[2:]
            else:
                tw, th = (8*len(text), 16)
            pad = 2
            draw.rectangle([x1, y1 - th - 2*pad, x1 + tw + 2*pad, y1], fill=color)
            draw.text((x1 + pad, y1 - th - pad), text, fill=(255,255,255), font=font)

    return vis

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Overlay Mask R-CNN detections on test images and save to detected/")
    ap.add_argument("--weights", type=str, default="cuts_maskrcnn_best.pth", help="Path to .pth weights")
    ap.add_argument("--sim_coco_dir", type=str, default="simulation_coco", help="Path to simulation_coco directory")
    ap.add_argument("--out_dir", type=str, default=None, help="Output dir (default: simulation_coco/test/detected)")
    ap.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--mask", type=float, default=0.5, help="Mask binarization threshold")
    ap.add_argument("--show", action="store_true", help="Preview a grid (requires matplotlib)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only this many images")
    ap.add_argument("--boxes", action="store_true", help="Also draw boxes + labels on top of masks")
    args = ap.parse_args()

    root = Path(__file__).parent.resolve()
    weights_path = (root / args.weights).resolve()
    sim_coco_dir = (root / args.sim_coco_dir).resolve()

    if not weights_path.exists():
        print(f"[error] Weights not found: {weights_path}")
        sys.exit(1)
    if not sim_coco_dir.exists():
        print(f"[error] simulation_coco not found: {sim_coco_dir}")
        sys.exit(1)

    test_images = find_test_images(sim_coco_dir)
    if not test_images:
        print(f"[error] No test images in {sim_coco_dir}/test or /test/images")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (sim_coco_dir / "test" / "detected")
    ensure_outdir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    model, class_names = load_detector(weights_path, device)
    print(f"[info] Loaded model; classes: {class_names}")

    saved_paths = []
    for i, img_path in enumerate(test_images):
        img_pil, preds = run_inference_on_image(
            model, img_path, device,
            conf_thresh=args.conf, iou_thresh=args.iou, mask_thresh=args.mask
        )
        vis = overlay_masks_and_boxes(img_pil, preds, class_names, draw_boxes=args.boxes)

        out_path = out_dir / img_path.name
        vis.save(out_path)
        saved_paths.append(out_path)
        print(f"[saved] {out_path}  (instances={preds['masks'].shape[0]})")

        if args.limit and (i + 1) >= args.limit:
            break

    if args.show:
        if not _HAVE_MPL:
            print("[warn] Matplotlib not available; skip preview.")
            return
        # Show up to 8 images in a grid
        n = min(8, len(saved_paths))
        cols = 4
        rows = (n + cols - 1) // cols
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        for i, p in enumerate(saved_paths[:n], 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(p.name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
