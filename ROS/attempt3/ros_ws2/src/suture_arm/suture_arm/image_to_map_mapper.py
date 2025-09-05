#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiny helper used by vision_web.py

Exports:
  - load_pad_mapper(calib_path: Optional[str]) -> Optional[PadMapper]
  - get_px_per_mm(mapper: Optional[PadMapper], img_w: int, img_h: int) -> float

If no JSON calibration is provided, we fall back to the training-mat heuristic
(≈120 mm × 175 mm) based on the current crop size.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json
import os


@dataclass
class PadMapper:
    """
    Optional calibration describing the suture pad / mat in the image.

    Expected JSON keys (all optional):
      {
        "mat_mm_w": 120.0,      # physical width (mm)
        "mat_mm_h": 175.0,      # physical height (mm)
        "px_per_mm": 6.4        # fixed px/mm if you already measured it
      }
    """
    cfg: Dict[str, Any]

    def get_px_per_mm(self, img_w: int, img_h: int) -> float:
        # 1) If explicit px/mm is provided, use it.
        val = self.cfg.get("px_per_mm", None)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)

        # 2) Otherwise infer using provided physical mat size (defaults 120×175 mm).
        mm_w = float(self.cfg.get("mat_mm_w", 120.0))
        mm_h = float(self.cfg.get("mat_mm_h", 175.0))

        ppm_w = float(img_w) / mm_w if (img_w and mm_w > 0) else 0.0
        ppm_h = float(img_h) / mm_h if (img_h and mm_h > 0) else 0.0

        # Average both directions; clamp to a sane positive value
        ppm = (ppm_w + ppm_h) * 0.5 if (ppm_w > 0 and ppm_h > 0) else max(ppm_w, ppm_h)
        return float(max(0.1, ppm))


def load_pad_mapper(calib_path: Optional[str]) -> Optional[PadMapper]:
    """
    Load calibration JSON if available. Returns None on any problem (vision_web
    will fall back to heuristic).
    """
    if not calib_path:
        return None
    try:
        with open(calib_path, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("pad mapper JSON must be a dict")
        return PadMapper(cfg)
    except Exception:
        return None

# image_to_map_mapper.py

def get_px_per_mm(mapper, img_w, img_h=None, fallback=None):
    """
    Returns pixels-per-millimeter. Accepts either:
      - img_w, img_h as numbers, OR
      - img_w=(H, W) tuple/list (common shape ordering), with img_h=None
    If no mapper calib is available, falls back to mat size (120x175 mm) or `fallback`.
    """
    # Accept (H, W) in a single tuple/list
    if img_h is None and isinstance(img_w, (tuple, list)) and len(img_w) == 2:
        H, W = img_w
        img_w, img_h = int(W), int(H)
    else:
        img_w = int(img_w)
        img_h = int(img_h) if img_h is not None else 0

    # Defaults if mapper missing
    mm_w = 120.0
    mm_h = 175.0
    if mapper:
        try:
            mm_w = float(mapper.get('mat_mm', {}).get('w', mm_w))
            mm_h = float(mapper.get('mat_mm', {}).get('h', mm_h))
        except Exception:
            pass

    ppm_w = (img_w / mm_w) if (img_w > 0 and mm_w > 0) else 0.0
    ppm_h = (img_h / mm_h) if (img_h > 0 and mm_h > 0) else 0.0
    ppm   = (ppm_w + ppm_h) * 0.5 if (ppm_w > 0 and ppm_h > 0) else max(ppm_w, ppm_h)

    if ppm <= 0.0:
        try:
            return float(fallback) if fallback is not None else 1.0
        except Exception:
            return 1.0
    return float(ppm)
