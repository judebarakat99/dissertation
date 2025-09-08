#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont


TORCH_OK = True
_TORCH_ERR = ""

def _pick_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return None


def _guess_wants_masks(weights_path: str, ckpt: dict) -> bool:
    name = weights_path.lower()
    if "mask" in name or "maskrcnn" in name:
        return True
    state = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt
    try:
        return any("mask" in k.lower() for k in state.keys())
    except Exception:
        return False


def _build_fallback_model(num_classes: int, want_masks: bool):
    if want_masks:
        return torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None, weights_backbone="IMAGENET1K_V1", num_classes=num_classes
        )
    else:
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone="IMAGENET1K_V1", num_classes=num_classes
        )


def load_detector(weights_path: str):
    """Return (model, class_names, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location="cpu")

    class_names = None
    if isinstance(ckpt, dict):
        for key in ("classes", "class_names", "names", "labels"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                class_names = list(ckpt[key])
                break
        if class_names is None:
            cti = ckpt.get("class_to_idx")
            if isinstance(cti, dict) and len(cti) > 0:
                inv = sorted(cti.items(), key=lambda kv: int(kv[1]))
                class_names = [k for k, _ in inv]

    num_classes = 2
    if class_names:
        num_classes = len(class_names) if class_names[0].lower() in ("__background__", "background", "bg") else len(class_names) + 1
    else:
        class_names = ["__background__", "cut"]

    want_masks = _guess_wants_masks(weights_path, ckpt if isinstance(ckpt, dict) else {})
    model = _build_fallback_model(num_classes=num_classes, want_masks=want_masks)

    state = ckpt.get("model", None) or ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None) or ckpt
    try:
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    return model, class_names, device


MASK_BIN_THR = 0.5

def _to_numpy_uint8_mask(m: "torch.Tensor") -> np.ndarray:
    t = torch.as_tensor(m)
    if t.dtype == torch.bool:
        t = t.to(dtype=torch.uint8) * 255
        if t.dim() == 2: t = t.unsqueeze(0)
        if t.dim() == 4 and t.shape[1] == 1: t = t[:, 0]
        return t.detach().cpu().numpy()
    if t.dtype.is_floating_point and (t.min() < 0 or t.max() > 1):
        t = t.sigmoid()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 4:
        if t.shape[1] == 1:
            t = t[:, 0]
        else:
            t, _ = t.max(dim=1)
    t = (t >= MASK_BIN_THR).to(dtype=torch.uint8) * 255
    return t.detach().cpu().numpy()


@torch.inference_mode()
def run_inference(model, device, frame_bgr: np.ndarray, conf_thresh: float = 0.4, iou_thresh: float = 0.5):
    """Return (img_pil, boxes, scores, labels, masks_np)."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tens = F.to_tensor(img_pil).to(device)

    out = model([tens])
    outputs = out[0] if isinstance(out, (list, tuple)) else out

    is_dict = isinstance(outputs, dict)
    boxes = outputs.get("boxes", None) if is_dict else None
    scores = outputs.get("scores", None) if is_dict else None
    labels = outputs.get("labels", None) if is_dict else None

    masks = None
    if is_dict:
        for k in ("masks", "pred_masks", "mask", "segmentation", "segm", "probs_mask"):
            if k in outputs:
                masks = outputs[k]
                break
    try:
        if masks is None and hasattr(outputs, "get_fields"):
            fields = outputs.get_fields()
            if "pred_masks" in fields:
                masks = fields["pred_masks"]
            boxes = boxes if boxes is not None else fields.get("pred_boxes", None)
            scores = scores if scores is not None else fields.get("scores", None)
            labels = labels if labels is not None else fields.get("pred_classes", None)
    except Exception:
        pass

    def _to_tensor(x, default_shape=()):
        try:
            return torch.as_tensor(x)
        except Exception:
            try:
                return torch.as_tensor(x.tensor)
            except Exception:
                return torch.empty(default_shape)

    boxes = _to_tensor(boxes, (0, 4))
    scores = _to_tensor(scores, (0,))
    labels = _to_tensor(labels, (0,)).to(dtype=torch.long)

    keep = (scores >= conf_thresh)
    if keep.numel() > 0:
        boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
        if masks is not None:
            try:
                masks = masks[keep]
            except Exception:
                if isinstance(masks, (list, tuple)):
                    masks = [m for m, k in zip(masks, keep.tolist()) if k]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_idx]; scores = scores[keep_idx]; labels = labels[keep_idx]
        if masks is not None:
            try:
                masks = masks[keep_idx]
            except Exception:
                if isinstance(masks, (list, tuple)):
                    masks = [masks[i] for i in keep_idx.tolist()]

    masks_np = None
    if masks is not None:
        try:
            masks_np = _to_numpy_uint8_mask(masks)
        except Exception as e:
            print(f"[model_infer][WARN] mask extraction failed: {e}", flush=True)
            masks_np = None

    return img_pil, boxes.cpu(), scores.cpu(), labels.cpu(), masks_np


def draw_dets(img_pil, boxes, scores, labels, class_names, selected_index=None, masks=None,
              poly_color=(0, 220, 220), poly_width=3):
    draw = ImageDraw.Draw(img_pil)
    font = _pick_font(16)

    # masks → contour
    if masks is not None:
        import cv2  # local to avoid global dep here
        import numpy as np
        is_torch = False
        try:
            import torch as _t
            is_torch = isinstance(masks, _t.Tensor)
        except Exception:
            pass
        if is_torch:
            M = masks.shape[0]
            def get_mask(k): return masks[k, 0].detach().cpu().numpy() if masks.ndim == 4 else masks[k].detach().cpu().numpy()
        else:
            M = masks.shape[0]
            def get_mask(k): return masks[k]
        for idx in range(M):
            m = get_mask(idx)
            mb = (m >= 128).astype(np.uint8)
            cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (255, 200, 30) if (selected_index is not None and idx == selected_index) else poly_color
            for c in cnts:
                if len(c) < 3:
                    continue
                poly = [(int(p[0][0]), int(p[0][1])) for p in c]
                draw.line(poly + [poly[0]], fill=color, width=int(poly_width))

    W, H = img_pil.size
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        w = x2 - x1; h = y2 - y1
        pad_w = 0.22 * w * 2.0
        pad_h = 0.08 * h
        vx1 = max(0.0, x1 - pad_w); vy1 = max(0.0, y1 - pad_h)
        vx2 = min(float(W), x2 + pad_w); vy2 = min(float(H), y2 + pad_h)

        color = (80, 180, 255) if (selected_index is not None and idx == selected_index) else (60, 200, 140)
        width = 5 if (selected_index is not None and idx == selected_index) else 3
        draw.rectangle([vx1, vy1, vx2, vy2], outline=color, width=width)

        lid = int(label)
        cls_name = class_names[lid] if 0 <= lid < len(class_names) else f"id_{lid}"
        text = f"{cls_name} {float(score):.2f}"
        if font: tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        else:    tw, th = (8*len(text), 14)
        pad = 2
        draw.rectangle([vx1, vy1 - th - 2*pad, vx1 + tw + 2*pad, vy1], fill=color)
        draw.text((vx1 + pad, vy1 - th - pad), text, fill=(255, 255, 255), font=font)
        idx_text = f"#{idx}"
        if font: itw, ith = draw.textbbox((0, 0), idx_text, font=font)[2:]
        else:    itw, ith = (12, 12)
        tag_pad = 2
        draw.rectangle([vx1, vy1, vx1 + itw + 2*tag_pad, vy1 + ith + 2*tag_pad], fill=(0, 0, 0))
        draw.text((vx1 + tag_pad, vy1 + tag_pad), idx_text, fill=(255, 255, 255), font=font)
    return img_pil


import cv2 
