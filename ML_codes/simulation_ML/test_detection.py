#!/usr/bin/env python3
# run_detection.py
import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
def find_test_images(sim_coco_dir: Path) -> List[Path]:
    cand_dirs = [
        sim_coco_dir / "test" / "images",     # common COCO export
        sim_coco_dir / "test"                 # flat folder
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
        # DejaVuSans is commonly available; fallback to default if missing
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return None

# ----------------------------
# Model loading
# ----------------------------
def _try_import_training_model() -> Optional[object]:
    """
    Try to import a model-constructor from your training script.
    We attempt a few common function names.
    """
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    try:
        import train_cuts_detector as tcd  # located next to weights
    except Exception:
        return None

    candidates = [
        "get_model",
        "create_model",
        "build_model",
        "make_model",
        "get_detector",
        "create_detector",
    ]
    for name in candidates:
        fn = getattr(tcd, name, None)
        if callable(fn):
            return fn
    return None

def build_fallback_model(num_classes: int = 2):
    """
    Sensible fallback: Faster R-CNN w/ ResNet50 FPN (pretrained backbone).
    Adjust if your project used a different architecture.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone="IMAGENET1K_V1",
        num_classes=num_classes
    )
    return model

def load_detector(weights_path: Path, device: torch.device):
    """
    Load the detector and return (model, class_names, model_kind).
    model_kind is 'torchvision' for standard detection models.
    """
    ckpt = torch.load(str(weights_path), map_location="cpu")

    # Guess class names if present in checkpoint metadata
    class_names = None
    if isinstance(ckpt, dict):
        for key in ("classes", "class_names", "names", "labels"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                class_names = list(ckpt[key])
                break

    # Try to reconstruct via training module (most robust)
    ctor = _try_import_training_model()
    model = None
    if ctor:
        try:
            # Common patterns: ctor(num_classes=...), or ctor()
            if class_names:
                # include background if not present
                if class_names and class_names[0].lower() not in ("__background__", "background", "bg"):
                    num_classes = len(class_names) + 1
                else:
                    num_classes = len(class_names)
            else:
                num_classes = 2
            try:
                model = ctor(num_classes=num_classes)
            except TypeError:
                model = ctor()
        except Exception as e:
            print(f"[warn] Could not build model via train_cuts_detector.*: {e}")

    if model is None:
        # Fallback model
        num_classes = len(class_names) + (0 if (class_names and class_names and class_names[0].lower() in ("__background__", "background", "bg")) else 1) if class_names else 2
        model = build_fallback_model(num_classes=num_classes)

    # Load weights
    state = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt
    try:
        # Remove possible "module." prefixes
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[warn] Non-strict load or key mismatch: {e}. Attempting strict=False...")
        model.load_state_dict(state, strict=False)

    model.eval().to(device)

    # Default class names if unknown
    if class_names is None:
        class_names = ["__background__", "cut"]

    return model, class_names, "torchvision"

# ----------------------------
# Inference + Visualization
# ----------------------------
@torch.inference_mode()
def run_inference_on_image(
    model,
    image_path: Path,
    device: torch.device,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5
):
    img_pil = Image.open(str(image_path)).convert("RGB")
    img_tensor = F.to_tensor(img_pil).to(device)

    outputs = model([img_tensor])[0]  # torchvision detection API

    boxes = outputs.get("boxes", torch.empty((0, 4), device=device))
    scores = outputs.get("scores", torch.empty((0,), device=device))
    labels = outputs.get("labels", torch.empty((0,), dtype=torch.long, device=device))

    # Confidence filter
    keep = scores >= conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # NMS
    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    return img_pil, boxes.cpu(), scores.cpu(), labels.cpu()

def draw_detections(
    img_pil: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
):
    draw = ImageDraw.Draw(img_pil)
    font = pick_font(16)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        # Pick color based on label id in a repeatable way
        color = tuple(int((hash(int(label)) >> (i * 8)) & 255) for i in range(3))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_idx = int(label)
        try:
            # If class_names includes background at index 0, this maps correctly
            cls_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else f"id_{label_idx}"
        except Exception:
            cls_name = f"id_{label_idx}"

        text = f"{cls_name} {score:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        pad = 2
        # caption background
        draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color)
        draw.text((x1 + pad, y1 - th - pad), text, fill=(255, 255, 255), font=font)
    return img_pil

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run detection on test images and save to detected/")
    parser.add_argument("--weights", type=str, default="cuts_detector_best.pth", help="Path to the .pth weights")
    parser.add_argument("--sim_coco_dir", type=str, default="simulation_coco", help="Path to simulation_coco directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: simulation_coco/test/detected)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--show", action="store_true", help="Show a quick preview window (matplotlib)")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many images")
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    weights_path = (root / args.weights).resolve()
    sim_coco_dir = (root / args.sim_coco_dir).resolve()

    if not weights_path.exists():
        print(f"[error] Weights not found at: {weights_path}")
        sys.exit(1)
    if not sim_coco_dir.exists():
        print(f"[error] simulation_coco directory not found at: {sim_coco_dir}")
        sys.exit(1)

    test_images = find_test_images(sim_coco_dir)
    if not test_images:
        print(f"[error] No test images found in {sim_coco_dir}/test or {sim_coco_dir}/test/images")
        sys.exit(1)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = sim_coco_dir / "test" / "detected"
    ensure_outdir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    model, class_names, model_kind = load_detector(weights_path, device)
    print(f"[info] Loaded model ({model_kind}); classes: {class_names}")

    # Optional preview
    preview_paths = []
    count = 0
    for img_path in test_images:
        img_pil, boxes, scores, labels = run_inference_on_image(
            model, img_path, device, conf_thresh=args.conf, iou_thresh=args.iou
        )
        vis = draw_detections(img_pil, boxes, scores, labels, class_names)
        out_path = out_dir / img_path.name
        vis.save(out_path)
        preview_paths.append(out_path)
        count += 1
        print(f"[saved] {out_path}  (n={len(boxes)})")
        if args.limit and count >= args.limit:
            break

    if args.show:
        # Show up to 8 images as a quick preview
        show_n = min(8, len(preview_paths))
        cols = 4
        rows = (show_n + cols - 1) // cols
        fig = plt.figure(figsize=(4 * cols, 4 * rows))
        for i, p in enumerate(preview_paths[:show_n], 1):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(p.name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
