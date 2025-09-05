#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vision_web.py
-------------
Flask web UI for CoppeliaSim vision sensors with (optional) ML detection and
suture-path publishing to ROS 2.

Endpoints expected by your existing index.html:
  GET  /                    -> template (index.html)
  GET  /stream_raw/<which>  -> MJPEG from "top" or "side"
  GET  /snapshot.jpg        -> snapshot (optionally with det overlays)
  GET  /detections          -> JSON of last detections
  POST /select_cut          -> select a detection id
  GET  /selected_crop.jpg   -> crop image for selected detection
  GET  /mask.jpg            -> crop mask image (grayscale)
  GET  /pattern.jpg         -> pattern overlay image
  POST /move_robot          -> publish /suture_cuts (std_msgs/String JSON)
  GET  /health              -> status flags

Optional calibration file (JSON) via env SUTURE_PAD_CALIB:
  - If provided and valid, we use its 'px_per_mm' (float) for px<->mm.
  - Otherwise we fall back to a heuristic: avg(W/120, H/175) from the crop.
"""

from __future__ import annotations

import io
import os
import time
import json
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from flask import (
    Flask, Response, render_template, abort, make_response,
    request, redirect, url_for, jsonify
)
from ament_index_python.packages import get_package_share_directory

# ---------- CoppeliaSim ZMQ remote API ----------
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy name

# ---------- Torch / torchvision (optional) ----------
TORCH_OK = True
_TORCH_ERR = ""
try:
    import torch
    import torchvision
    from torchvision.ops import nms
    from torchvision.transforms import functional as F
    from PIL import Image, ImageDraw, ImageFont
except Exception as _e:
    TORCH_OK = False
    _TORCH_ERR = str(_e)
    # light shims to avoid NameErrors when TORCH_OK is False
    Image = None
    ImageDraw = None
    ImageFont = None
    def nms(*_, **__): return []
    def F(*_, **__): return None  # not used when TORCH_OK is False

# ---------- Stitching (pattern) helpers ----------
try:
    from . import stitching
except Exception:
    import stitching


# ====================== Configuration ======================

CSIM_HOST = os.getenv("CSIM_HOST", "127.0.0.1")
CSIM_PORT = int(os.getenv("CSIM_PORT", "23000"))

TOP_SENSOR_ALIAS = os.getenv("TOP_SENSOR_ALIAS", "/visionSensor")
SIDE_SENSOR_ALIAS = os.getenv("SIDE_SENSOR_ALIAS", "/visionSensor_SideView")

FPS = int(os.getenv("VISION_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))

STEPPED_SNAPSHOT = int(os.getenv("STEPPED_SNAPSHOT", "1"))  # step sim between snapshots
STEPPED_RAW = int(os.getenv("STEPPED_RAW", "0"))            # usually leave raw stream continuous

# Optional calibration file (JSON) for px/mm etc.
CALIB_PATH: str = os.getenv("SUTURE_PAD_CALIB", "").strip()
PAD_MAPPER: Optional[dict] = None  # loaded JSON, if any


# ====================== Templates & models ======================

def _find_templates_dir() -> str:
    env = os.getenv("SUTURE_ARM_TEMPLATES", "").strip()
    if env and os.path.isdir(env):
        return env
    # package share
    try:
        share = get_package_share_directory("suture_arm")
        tdir = os.path.join(share, "templates")
        if os.path.isdir(tdir):
            return tdir
    except Exception:
        pass
    # dev fallback
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "templates"))


def _find_models_root() -> str:
    # Prefer explicit env
    env = os.getenv("SUTURE_ARM_ML", "").strip()
    if env and os.path.isdir(env):
        return env
    # package share default
    try:
        share = get_package_share_directory("suture_arm")
        mdir = os.path.join(share, "ml")
        if os.path.isdir(mdir):
            return mdir
    except Exception:
        pass
    # dev fallback
    here = os.path.dirname(__file__)
    cand = os.path.abspath(os.path.join(here, "..", "ML_detection"))
    return cand if os.path.isdir(cand) else here


TEMPLATES_DIR = _find_templates_dir()
MODELS_ROOT = _find_models_root()

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    raise RuntimeError(
        f"index.html not found in {TEMPLATES_DIR}. "
        "Place it under src/suture_arm/templates/ or set SUTURE_ARM_TEMPLATES."
    )

app = Flask(__name__, template_folder=TEMPLATES_DIR)


# ====================== Calibration loader ======================

def load_pad_mapper():
    """Load optional calibration JSON from CALIB_PATH (if provided)."""
    global PAD_MAPPER
    PAD_MAPPER = None

    if not CALIB_PATH:
        print("[vision_web] pad mapper not set; using px/mm heuristic fallback.", flush=True)
        return

    if not os.path.isfile(CALIB_PATH):
        print(f"[vision_web] pad mapper '{CALIB_PATH}' not found; using px/mm heuristic fallback.", flush=True)
        return

    try:
        with open(CALIB_PATH, "r") as f:
            PAD_MAPPER = json.load(f)
        print(f"[vision_web] loaded pad mapper: {CALIB_PATH}", flush=True)
    except Exception as e:
        PAD_MAPPER = None
        print(f"[vision_web] failed to load pad mapper '{CALIB_PATH}': {e}. Using px/mm heuristic fallback.", flush=True)


def get_px_per_mm_from_mapper(default_ppm: float) -> float:
    """Return px_per_mm from PAD_MAPPER if available; else default."""
    try:
        if isinstance(PAD_MAPPER, dict):
            v = PAD_MAPPER.get("px_per_mm", None)
            if v is not None and float(v) > 0:
                return float(v)
    except Exception:
        pass
    return float(default_ppm)


# ====================== ROS 2 publisher (/suture_cuts) ======================

cuts_node = None
cuts_pub = None

def init_ros2_publisher():
    """Initialize a persistent ROS2 publisher for /suture_cuts."""
    global cuts_node, cuts_pub
    try:
        import rclpy
        from rclpy.node import Node as _RclNode
        from std_msgs.msg import String as _MsgString
    except Exception as e:
        print(f"[vision_web][ERR] ROS2 modules not available: {e}", flush=True)
        return

    try:
        if not rclpy.ok():
            rclpy.init(args=None)

        class _CutsNode(_RclNode):
            def __init__(self):
                super().__init__('suture_cuts_web_publisher')
                self.pub = self.create_publisher(_MsgString, '/suture_cuts', 10)

        cuts_node = _CutsNode()
        cuts_pub = cuts_node.pub
        print("[vision_web] ROS2 publisher ready on /suture_cuts", flush=True)

        # OPTIONAL: keep the graph warm with a tiny spin_once in a background thread
        def _spin_trickle():
            while rclpy.ok():
                try:
                    rclpy.spin_once(cuts_node, timeout_sec=0.05)
                except Exception:
                    break
                time.sleep(0.20)
        th = threading.Thread(target=_spin_trickle, daemon=True)
        th.start()

    except Exception as e:
        cuts_node = None
        cuts_pub = None
        print(f"[vision_web][ERR] ROS2 publisher init failed: {e}", flush=True)


# ====================== CoppeliaSim frame grabbing ======================

def _decode_buffer_to_bgr(img, w: int, h: int) -> np.ndarray:
    buf = np.frombuffer(img, dtype=np.uint8) if isinstance(img, (bytes, bytearray)) else np.array(img, dtype=np.uint8)
    n = buf.size
    if n == w * h:
        frame = cv2.cvtColor(np.flip(buf.reshape(h, w), 0), cv2.COLOR_GRAY2BGR)
    elif n == w * h * 3:
        frame = np.flip(buf.reshape(h, w, 3), 0)[:, :, ::-1]
    elif n == w * h * 4:
        frame = np.flip(buf.reshape(h, w, 4), 0)[:, :, :3][:, :, ::-1]
    else:
        if (w * h) and n % (w * h) == 0:
            c = n // (w * h)
            frame = np.flip(buf.reshape(h, w, c), 0)[:, :, :3][:, :, ::-1]
        else:
            raise RuntimeError(f"unexpected buffer size n={n} vs {w}x{h}")
    if not (frame.flags["C_CONTIGUOUS"] and frame.flags["WRITEABLE"]):
        frame = np.ascontiguousarray(frame.copy())
    return frame


class FrameGrabber:
    """Background thread that keeps the latest frame from a vision sensor."""

    def __init__(self, host: str, port: int, sensor_alias: str):
        self.host = host
        self.port = port
        self.sensor_alias = sensor_alias
        self.client = None
        self.sim = None
        self.sensor = None
        self._last = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _resolve(self, alias: str) -> Optional[int]:
        for cand in (alias, alias.lstrip("/"), alias + "#0"):
            try:
                return self.sim.getObject(cand)
            except Exception:
                pass
        return None

    def _connect(self):
        self.client = RemoteAPIClient(self.host, self.port)
        self.sim = self.client.require("sim")
        self.sensor = self._resolve(self.sensor_alias)
        if self.sensor is None:
            raise RuntimeError(f"Vision sensor '{self.sensor_alias}' not found")
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass
        if STEPPED_SNAPSHOT and hasattr(self.sim, "setStepping"):
            try:
                self.sim.setStepping(True)
            except Exception:
                pass

    def _read_frame(self) -> np.ndarray:
        try:
            if bool(self.sim.getObjectInt32Param(self.sensor, self.sim.visionintparam_explicit_handling)):
                self.sim.handleVisionSensor(self.sensor)
        except Exception:
            pass
        if STEPPED_SNAPSHOT and hasattr(self.sim, "step"):
            try:
                self.sim.step()
            except Exception:
                pass
        try:
            img, res = self.sim.getVisionSensorImg(self.sensor)  # bytes, [w,h]
            w, h = int(res[0]), int(res[1])
        except Exception as e1:
            try:
                out = self.sim.getVisionSensorImage(self.sensor)
                if isinstance(out, (list, tuple)) and len(out) == 3 and isinstance(out[1], (int, float)):
                    img, w, h = out
                    w, h = int(w), int(h)
                else:
                    img, res = out
                    w, h = int(res[0]), int(res[1])
                if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
                    arr = np.array(img, dtype=np.float32)
                    img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
            except Exception as e2:
                raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")
        return _decode_buffer_to_bgr(img, w, h)

    def _loop(self):
        period = 1.0 / max(1, FPS)
        while not self._stop.is_set():
            try:
                if self.sim is None:
                    self._connect()
                frame = self._read_frame()
                with self._lock:
                    self._last = (frame, time.time())
            except Exception as e:
                print(f"[vision_web][ERR] snapshot capture: {e}", flush=True)
                time.sleep(0.1)
            time.sleep(max(0.0, period * 0.5))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last is None:
                return None
            return self._last[0].copy()


grabber = FrameGrabber(CSIM_HOST, CSIM_PORT, TOP_SENSOR_ALIAS)


# MJPEG (raw) helper state
_raw_ctx: Dict[str, Dict] = {}

def _raw_connect(alias: str):
    if alias in _raw_ctx and _raw_ctx[alias].get("sim") and _raw_ctx[alias].get("sensor"):
        return
    client = RemoteAPIClient(CSIM_HOST, CSIM_PORT)
    sim = client.require("sim")
    sensor = None
    for cand in (alias, alias.lstrip("/"), alias + "#0"):
        try:
            sensor = sim.getObject(cand)
            break
        except Exception:
            pass
    if sensor is None:
        raise RuntimeError(f"RAW: vision sensor '{alias}' not found")
    try:
        st = sim.getSimulationState()
        if st in (sim.simulation_stopped, sim.simulation_paused):
            sim.startSimulation()
    except Exception:
        pass
    _raw_ctx[alias] = {"client": client, "sim": sim, "sensor": sensor}


def _raw_read_once(alias: str) -> np.ndarray:
    ctx = _raw_ctx[alias]
    sim = ctx["sim"]
    sensor = ctx["sensor"]
    try:
        if bool(sim.getObjectInt32Param(sensor, sim.visionintparam_explicit_handling)):
            sim.handleVisionSensor(sensor)
    except Exception:
        pass
    try:
        img, res = sim.getVisionSensorImg(sensor)  # bytes, [w,h]
        w, h = int(res[0]), int(res[1])
    except Exception:
        out = sim.getVisionSensorImage(sensor)
        if isinstance(out, (list, tuple)) and len(out) == 3 and isinstance(out[1], (int, float)):
            img, w, h = out
            w, h = int(w), int(h)
        else:
            img, res = out
            w, h = int(res[0]), int(res[1])
        if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
            arr = np.array(img, dtype=np.float32)
            img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
    return _decode_buffer_to_bgr(img, w, h)


def encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def mjpeg_generator_raw(alias: str):
    try:
        _raw_connect(alias)
    except Exception as e:
        msg = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(msg, f"RAW connect error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_jpeg(msg) + b"\r\n")
        return
    period = 1.0 / max(1, FPS)
    while True:
        try:
            frame = _raw_read_once(alias)
            jpg = encode_jpeg(frame)
        except Exception as e:
            err = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(err, f"RAW error: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            jpg = encode_jpeg(err)
            time.sleep(0.1)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(period)


# ====================== Model (detection) ======================

def _pick_font(size: int = 16):
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return None


def _guess_wants_masks(weights_path: str, ckpt: dict) -> bool:
    name = os.path.basename(weights_path).lower()
    if "mask" in name or "maskrcnn" in name:
        return True
    state = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt
    try:
        for k in state.keys():
            if "mask" in k.lower():
                return True
    except Exception:
        pass
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

    # Defaults
    if class_names:
        num_classes = len(class_names) if class_names[0].lower() in ("__background__", "background", "bg") else len(class_names) + 1
    else:
        class_names = ["__background__", "cut"]
        num_classes = 2

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
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() == 4 and t.shape[1] == 1:
            t = t[:, 0]
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
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if masks is not None:
            try:
                masks = masks[keep]
            except Exception:
                if isinstance(masks, (list, tuple)):
                    masks = [m for m, k in zip(masks, keep.tolist()) if k]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
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
            print(f"[vision_web][WARN] mask extraction failed: {e}", flush=True)
            masks_np = None

    return img_pil, boxes.cpu(), scores.cpu(), labels.cpu(), masks_np


def draw_dets(img_pil, boxes, scores, labels, class_names, selected_index=None, masks=None,
              poly_color=(0, 220, 220), poly_width=3):
    draw = ImageDraw.Draw(img_pil)
    font = _pick_font(16)

    # masks → contours
    if masks is not None:
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
        w = x2 - x1
        h = y2 - y1
        pad_w = 0.22 * w * 2.0
        pad_h = 0.08 * h
        vx1 = max(0.0, x1 - pad_w)
        vy1 = max(0.0, y1 - pad_h)
        vx2 = min(float(W), x2 + pad_w)
        vy2 = min(float(H), y2 + pad_h)

        color = (80, 180, 255) if (selected_index is not None and idx == selected_index) else (60, 200, 140)
        width = 5 if (selected_index is not None and idx == selected_index) else 3
        draw.rectangle([vx1, vy1, vx2, vy2], outline=color, width=width)

        lid = int(label)
        cls_name = class_names[lid] if 0 <= lid < len(class_names) else f"id_{lid}"
        text = f"{cls_name} {float(score):.2f}"
        if font:
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        else:
            tw, th = (8 * len(text), 14)
        pad = 2
        draw.rectangle([vx1, vy1 - th - 2 * pad, vx1 + tw + 2 * pad, vy1], fill=color)
        draw.text((vx1 + pad, vy1 - th - pad), text, fill=(255, 255, 255), font=font)

        idx_text = f"#{idx}"
        if font:
            itw, ith = draw.textbbox((0, 0), idx_text, font=font)[2:]
        else:
            itw, ith = (12, 12)
        tag_pad = 2
        draw.rectangle([vx1, vy1, vx1 + itw + 2 * tag_pad, vy1 + ith + 2 * tag_pad], fill=(0, 0, 0))
        draw.text((vx1 + tag_pad, vy1 + tag_pad), idx_text, fill=(255, 255, 255), font=font)
    return img_pil


# ====================== Model selection / state ======================

def _available_models() -> List[str]:
    try:
        files = [f for f in os.listdir(MODELS_ROOT) if f.lower().endswith(".pth")]
        # prefer cuts_maskrcnn_best
        files.sort(key=lambda n: (0 if "cuts_maskrcnn_best" in n.lower() else 1, n.lower()))
        return files
    except Exception:
        return []


def _select_weights_file() -> Optional[str]:
    models = _available_models()
    if not models:
        return None
    return os.path.join(MODELS_ROOT, models[0])


DETECTOR = None
WEIGHTS_PATH = None
LAST_DETS: Optional[Dict] = None
LAST_FRAME_PIL: Optional["Image.Image"] = None
SELECTED_DET_ID: Optional[int] = None
LAST_MASKS: Optional[List[np.ndarray]] = None


# ====================== Routes ======================

@app.route("/")
def index():
    model_name = os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else "(none)"
    return render_template(
        "index.html",
        top_sensor=TOP_SENSOR_ALIAS,
        side_sensor=SIDE_SENSOR_ALIAS,
        host=CSIM_HOST,
        port=CSIM_PORT,
        model=model_name,
        models=_available_models(),
    )


@app.route("/select_model", methods=["POST"])
def select_model():
    fname = request.form.get("weights", "").strip()
    if not fname:
        return redirect(url_for("index"))
    candidate = os.path.join(MODELS_ROOT, fname)
    if not os.path.isfile(candidate):
        return redirect(url_for("index"))
    if not TORCH_OK:
        return redirect(url_for("index"))

    global DETECTOR, WEIGHTS_PATH, LAST_DETS, SELECTED_DET_ID, LAST_FRAME_PIL, LAST_MASKS
    try:
        DETECTOR = load_detector(candidate)
        WEIGHTS_PATH = candidate
        LAST_DETS = None
        LAST_FRAME_PIL = None
        LAST_MASKS = None
        SELECTED_DET_ID = None
    except Exception as e:
        print(f"[vision_web][ERR] failed to load {candidate}: {e}", flush=True)
    return redirect(url_for("index"))


@app.route("/stream_raw/<which>")
def stream_raw(which: str):
    alias = TOP_SENSOR_ALIAS if which == "top" else SIDE_SENSOR_ALIAS
    return Response(mjpeg_generator_raw(alias), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot_image():
    frame = grabber.get_frame()
    if frame is None:
        abort(503, "no frame yet")

    if (not TORCH_OK) or (DETECTOR is None):
        jpg = encode_jpeg(frame)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    hex_col = request.args.get("poly_color", default="#00dcdc", type=str)
    poly_width = request.args.get("poly_width", default=3, type=int)

    # parse hex to (r,g,b)
    def _parse_hex_color(s: Optional[str], default=(0, 220, 220)):
        if not s:
            return default
        try:
            s = s.strip()
            if s.startswith("#"):
                s = s[1:]
            if len(s) == 3:
                s = ''.join([ch * 2 for ch in s])
            if len(s) != 6:
                return default
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b)
        except Exception:
            return default

    poly_color = _parse_hex_color(hex_col, default=(0, 220, 220))

    model, class_names, device = DETECTOR
    try:
        img_pil, boxes, scores, labels, masks_np = run_inference(model, device, frame)

        global LAST_DETS, LAST_FRAME_PIL, LAST_MASKS
        cnames = class_names or ["__background__", "cut"]

        LAST_DETS = {
            "boxes": boxes.numpy().tolist(),
            "scores": scores.numpy().tolist(),
            "labels": labels.numpy().tolist(),
            "class_names": cnames,
            "image_size": [img_pil.width, img_pil.height],
            "ts": time.time(),
        }
        LAST_FRAME_PIL = img_pil.copy()
        LAST_MASKS = [m.copy() for m in masks_np] if (masks_np is not None and len(masks_np) > 0) else None

        vis = draw_dets(
            img_pil, boxes, scores, labels, cnames,
            selected_index=SELECTED_DET_ID, masks=masks_np,
            poly_color=poly_color, poly_width=int(poly_width)
        )
        buf = io.BytesIO()
        vis.save(buf, format="JPEG", quality=int(JPEG_QUALITY))
        jpg = buf.getvalue()
    except Exception as e:
        dbg = frame.copy()
        cv2.putText(dbg, f"inference error: {e}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        jpg = encode_jpeg(dbg)

    resp = make_response(jpg)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/detections")
def detections_json():
    if LAST_DETS is None:
        return jsonify({"detections": [], "message": "No snapshot run yet."})
    boxes = LAST_DETS.get("boxes", [])
    scores = LAST_DETS.get("scores", [])
    labels = LAST_DETS.get("labels", [])
    cnames = LAST_DETS.get("class_names", ["__background__", "cut"])
    dets = []
    for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(float, b)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        lab_name = cnames[int(l)] if (0 <= int(l) < len(cnames)) else f"id_{int(l)}"
        dets.append({
            "id": i, "score": float(s), "label": int(l), "label_name": lab_name,
            "box": [x1, y1, x2, y2], "cx": cx, "cy": cy, "w": w, "h": h, "area": w * h
        })
    return jsonify({
        "detections": dets,
        "selected_id": SELECTED_DET_ID,
        "model": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
        "ts": LAST_DETS.get("ts"),
        "size": LAST_DETS.get("image_size")
    })


@app.route("/select_cut", methods=["POST"])
def select_cut():
    global SELECTED_DET_ID
    try:
        idx = None
        if request.is_json:
            payload = request.get_json(force=True, silent=True) or {}
            idx = payload.get("id", None)
        else:
            idx = request.form.get("id", None)
        if idx is None:
            return jsonify({"ok": False, "error": "no id provided"}), 400
        idx = int(idx)
        n = len(LAST_DETS["boxes"]) if LAST_DETS else 0
        if idx < 0 or idx >= n:
            return jsonify({"ok": False, "error": f"id {idx} out of range (0..{max(0, n-1)})"}), 400
        SELECTED_DET_ID = idx
        return jsonify({"ok": True, "selected_id": SELECTED_DET_ID})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _get_selected_crop_and_mask_np(
    det_id: Optional[int],
    pad: float = 0.08,
    outw: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (crop_bgr, crop_mask) using asymmetric padding (wider horizontally)."""
    if LAST_DETS is None or LAST_FRAME_PIL is None:
        raise RuntimeError("No snapshot available. Click Snapshot first.")
    if det_id is None:
        det_id = SELECTED_DET_ID
    if det_id is None:
        raise RuntimeError("No detection selected.")
    boxes = LAST_DETS.get("boxes", [])
    if det_id < 0 or det_id >= len(boxes):
        raise RuntimeError(f"Selected id {det_id} out of range.")
    if LAST_MASKS is None or det_id >= len(LAST_MASKS) or LAST_MASKS[det_id] is None:
        raise RuntimeError("ML mask not available. Load cuts_maskrcnn_best.pth and refresh Snapshot.")

    x1, y1, x2, y2 = [float(v) for v in boxes[det_id]]
    W, H = LAST_DETS.get("image_size", [LAST_FRAME_PIL.width, LAST_FRAME_PIL.height])

    pad_w = float(request.args.get("pad_w", default=pad * 2.2))
    pad_h = float(request.args.get("pad_h", default=pad))
    pw = pad_w * (x2 - x1)
    ph = pad_h * (y2 - y1)

    xx1 = max(0, int(np.floor(x1 - pw))) - 10
    yy1 = max(0, int(np.floor(y1 - ph)))
    xx2 = 10 + min(int(np.ceil(x2 + pw)), int(W))
    yy2 = min(int(np.ceil(y2 + ph)), int(H))

    crop_pil = LAST_FRAME_PIL.crop((xx1, yy1, xx2, yy2))
    crop_rgb = np.array(crop_pil)
    if crop_rgb.ndim == 2:
        crop_rgb = np.repeat(crop_rgb[..., None], 3, axis=2)
    crop_bgr = crop_rgb[:, :, ::-1].copy()

    full = LAST_MASKS[det_id]  # HxW uint8
    crop_mask = full[yy1:yy2, xx1:xx2].copy()

    if outw and outw > 0:
        w, h = crop_pil.size
        outh = max(1, int(round(h * (outw / float(w)))))
        crop_bgr = cv2.resize(crop_bgr, (outw, outh), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(crop_mask, (outw, outh), interpolation=cv2.INTER_NEAREST)

    return crop_bgr, crop_mask


@app.route("/selected_crop.jpg")
def selected_crop():
    try:
        det_id = request.args.get("id", None, type=int)
        pad = request.args.get("pad", default=0.08, type=float)
        outw = request.args.get("size", default=None, type=int)
        crop_bgr, _ = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        jpg = encode_jpeg(crop_bgr)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((120, 480, 3), np.uint8)
        cv2.putText(err, f"crop err: {e}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        jpg = encode_jpeg(err)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


@app.route("/mask.jpg")
def mask_image():
    try:
        det_id = request.args.get("id", None, type=int)
        pad = request.args.get("pad", default=0.08, type=float)
        outw = request.args.get("size", default=None, type=int)
        _, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        vis = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
        jpg = encode_jpeg(vis, JPEG_QUALITY)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((160, 680, 3), np.uint8)
        cv2.putText(err, f"mask error: {e}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        jpg = encode_jpeg(err)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


@app.route("/pattern.jpg")
def pattern_image():
    try:
        det_id = request.args.get("id", None, type=int)
        pad = request.args.get("pad", default=0.08, type=float)
        outw = request.args.get("size", default=None, type=int)

        style = request.args.get("pattern", default="continuous", type=str).lower()
        spacing = request.args.get("spacing", default=20, type=int)
        curvature_gain = request.args.get("curvature_gain", default=18.0, type=float)
        outside_scale = request.args.get("outside_scale", default=0.12, type=float)
        border_push_px = request.args.get("border_push_px", default=0.0, type=float)
        bez_samples = request.args.get("bez_samples", default=16, type=int)
        thread_color_hex = request.args.get("thread_color", default="#1ed5ff", type=str)
        thread_thick = request.args.get("thread_thick", default=2, type=int)
        entry_mm = request.args.get("entry_mm", default=6.0, type=float)
        s_min = request.args.get("s_min", default=8, type=int)   # legacy
        s_max = request.args.get("s_max", default=60, type=int)  # legacy
        px_per_mm = request.args.get("px_per_mm", default=None, type=float)

        # clamp clinically requested 5–10 mm
        entry_mm = float(min(10.0, max(5.0, entry_mm)))

        # parse thread color (hex -> BGR for OpenCV)
        def _hex_to_bgr(s: str, default=(30, 200, 255)):
            try:
                s = s.strip()
                if s.startswith("#"): s = s[1:]
                if len(s) == 3: s = ''.join([c * 2 for c in s])
                if len(s) != 6: return default
                r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
                return (b, g, r)
            except Exception:
                return default

        thread_color = _hex_to_bgr(thread_color_hex)

        # crop + mask
        crop_bgr, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        if crop_mask is None or crop_mask.size == 0:
            crop_mask = np.zeros(crop_bgr.shape[:2], np.uint8)
        elif crop_mask.ndim == 3:
            crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

        # px/mm: explicit > mapper > heuristic
        if px_per_mm in (None, ""):
            H, W = crop_bgr.shape[:2]
            ppm_w = float(W) / 120.0
            ppm_h = float(H) / 175.0
            ppm_heur = max(0.0, (ppm_w + ppm_h) * 0.5)
            px_per_mm = get_px_per_mm_from_mapper(ppm_heur)

        max_probe = int(max(crop_mask.shape) * 2.0)
        common_kwargs = dict(
            spacing_px=float(spacing),
            max_probe=int(max_probe),
            color_thread=thread_color,
            thickness=int(thread_thick),
            debug=False,
            curvature_gain=float(curvature_gain),
            outside_scale=float(outside_scale),
            spur_min_px=4,
            entry_mm=float(entry_mm),
            px_per_mm=float(px_per_mm) if px_per_mm else None,
        )

        if style == "perp":
            over, _ = stitching.draw_stitching_pattern(
                crop_bgr, crop_mask, info={},
                spacing=int(spacing), length_scale=2.0,
                color=thread_color, thickness=int(thread_thick),
                curvature_gain=float(curvature_gain), outside_scale=float(outside_scale),
                spur_min_px=4, entry_mm=float(entry_mm), px_per_mm=float(px_per_mm) if px_per_mm else None,
            )
        elif style == "mold":
            over, _ = stitching.draw_stitching_mold_border(
                crop_bgr, crop_mask, grow_px=3,
                border_push_px=float(border_push_px),
                **common_kwargs
            )
        elif style == "bezier":
            fn = getattr(stitching, "draw_stitching_bezier", None)
            if fn is None:
                over, _ = stitching.draw_running_suture_auto(
                    crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
                )
            else:
                over, _ = fn(
                    crop_bgr, crop_mask,
                    outside_px=3.0, rect_min_step=6.0,
                    bezier_samples_per=int(bez_samples),
                    **common_kwargs
                )
        else:
            over, _ = stitching.draw_running_suture_auto(
                crop_bgr, crop_mask,
                outside_px=3.0, rect_min_step=6.0,
                **common_kwargs
            )

        jpg = encode_jpeg(over, JPEG_QUALITY)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        err = np.zeros((220, 980, 3), np.uint8)
        cv2.putText(err, f"pattern error: {e}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        jpg = encode_jpeg(err, JPEG_QUALITY)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


@app.route("/move_robot", methods=["POST"])
def move_robot():
    """
    Build a suture polyline from the selected cut (in crop pixels),
    convert to meters in the "mat" frame, and publish a JSON message on /suture_cuts.
    """
    print("[vision_web] /move_robot called", flush=True)

    # ROS message type (lazy import)
    try:
        from std_msgs.msg import String as _MsgString
    except Exception as e:
        return jsonify({"ok": False, "error": f"ROS2 std_msgs import failed: {e}"}), 500

    payload = {}
    if request.is_json:
        try:
            payload = request.get_json(force=True) or {}
        except Exception:
            payload = {}

    det_id = int(payload.get("det_id", SELECTED_DET_ID if SELECTED_DET_ID is not None else -1))
    style = str(payload.get("pattern", "continuous")).lower()
    spacing_px = float(payload.get("spacing", 20))
    entry_mm = float(payload.get("entry_mm", 6.0))
    depth_mm = float(payload.get("depth_mm", 3.0))
    pad = float(payload.get("pad", 0.08))
    outw = int(payload.get("size", 320))
    curvature_gain = float(payload.get("curvature_gain", 18.0))
    bez_samples = int(payload.get("bez_samples", 16))
    outside_scale = float(payload.get("outside_scale", 0.12))
    thread_thick = int(payload.get("thread_thick", 2))

    # clamp 5–10 mm
    entry_mm = max(5.0, min(10.0, entry_mm))

    if det_id < 0:
        return jsonify({"ok": False, "error": "No detection selected"}), 400

    # crop + mask
    try:
        crop_bgr, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
    except Exception as e:
        return jsonify({"ok": False, "error": f"crop/mask error: {e}"}), 400
    if crop_mask is None or crop_mask.size == 0:
        return jsonify({"ok": False, "error": "Empty mask/crop"}), 400
    if crop_mask.ndim == 3:
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

    max_probe = int(max(crop_mask.shape) * 2.0)
    thread_color = (30, 200, 255)
    common_kwargs = dict(
        spacing_px=float(spacing_px),
        max_probe=int(max_probe),
        color_thread=thread_color,
        thickness=int(thread_thick),
        debug=False,
        curvature_gain=float(curvature_gain),
        outside_scale=float(outside_scale),
        spur_min_px=4,
        entry_mm=float(entry_mm),
        px_per_mm=None,
    )

    # generate path in pixels
    try:
        if style == "perp":
            _, path2d = stitching.draw_stitching_pattern(
                crop_bgr, crop_mask, info={}, spacing=int(spacing_px), length_scale=2.0,
                color=thread_color, thickness=int(thread_thick),
                curvature_gain=float(curvature_gain), outside_scale=float(outside_scale),
                spur_min_px=4, entry_mm=float(entry_mm), px_per_mm=None,
            )
        elif style == "mold":
            _, path2d = stitching.draw_stitching_mold_border(
                crop_bgr, crop_mask, grow_px=3, border_push_px=0.0, **common_kwargs
            )
        elif style == "bezier":
            _, path2d = stitching.draw_stitching_bezier(
                crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0,
                bezier_samples_per=int(bez_samples), **common_kwargs
            )
        else:
            _, path2d = stitching.draw_running_suture_auto(
                crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
            )
    except Exception as e:
        return jsonify({"ok": False, "error": f"path gen error: {e}"}), 500

    if not path2d:
        return jsonify({"ok": False, "error": "No path produced from the selected cut"}), 400

    H, W = crop_bgr.shape[:2]

    # px/mm -> explicit from PAD_MAPPER if present, else heuristic from crop size
    ppm_w = float(W) / 120.0
    ppm_h = float(H) / 175.0
    ppm_heur = (ppm_w + ppm_h) * 0.5 if (ppm_w > 0 and ppm_h > 0) else 1.0
    px_per_mm = get_px_per_mm_from_mapper(ppm_heur)

    def px_to_m(u_px: float, v_px: float) -> Tuple[float, float]:
        # mm -> m (divide by 1000)
        return (float(u_px) / px_per_mm / 1000.0,
                float(v_px) / px_per_mm / 1000.0)

    poly_m = [px_to_m(u, v) for (u, v) in path2d]

    spacing_m = float(spacing_px) / px_per_mm / 1000.0
    entry_m = float(entry_mm) / 1000.0
    depth_m = float(depth_mm) / 1000.0

    cuts_msg = {
        "frame_id": "mat",
        "cuts": [{"polyline": [[x, y] for (x, y) in poly_m]}],
        "params": {
            "spacing": max(1e-4, spacing_m),
            "entry_mm": entry_m,  # meters (legacy key name)
            "depth": depth_m
        }
    }

    # publish
    global cuts_pub
    if cuts_pub is None:
        return jsonify({"ok": False, "error": "ROS2 publisher not initialized"}), 500

    from std_msgs.msg import String as _MsgString  # already checked above
    msg = _MsgString()
    msg.data = json.dumps(cuts_msg)
    cuts_pub.publish(msg)
    print(f"[vision_web] published /suture_cuts with {len(poly_m)} points", flush=True)

    return jsonify({
        "ok": True,
        "published": True,
        "topic": "/suture_cuts",
        "n_poly_pts": len(poly_m),
        "spacing_m": spacing_m,
        "entry_m": entry_m,
        "depth_m": depth_m
    })


@app.route("/health")
def health():
    ok = grabber.get_frame() is not None
    torch_ok = TORCH_OK
    model_loaded = DETECTOR is not None
    return (
        {
            "frame": ok,
            "torch": torch_ok,
            "model_loaded": model_loaded,
            "model_name": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
        },
        200 if ok else 503,
    )


# ====================== Main ======================

def main():
    print(f"[vision_web] templates dir: {TEMPLATES_DIR}", flush=True)
    print(f"[vision_web] models root  : {MODELS_ROOT}", flush=True)

    # Start grabber
    grabber.start()
    t0 = time.time()
    while grabber.get_frame() is None and (time.time() - t0) < 5.0:
        time.sleep(0.05)

    # Load calibration (safe)
    load_pad_mapper()

    # Init ROS2 pub
    init_ros2_publisher()

    # Load detector if available
    global DETECTOR, WEIGHTS_PATH
    if TORCH_OK:
        WEIGHTS_PATH = _select_weights_file()
        if WEIGHTS_PATH and os.path.isfile(WEIGHTS_PATH):
            try:
                DETECTOR = load_detector(WEIGHTS_PATH)
                print(f"[vision_web] loaded weights: {WEIGHTS_PATH}", flush=True)
            except Exception as e:
                print(f"[vision_web][ERR] failed to load detector: {e}", flush=True)
                DETECTOR = None
        else:
            print("[vision_web] no *.pth found under models root", flush=True)
    else:
        print(f"[vision_web][WARN] PyTorch unavailable: {_TORCH_ERR}", flush=True)

    print("[vision_web] open http://127.0.0.1:8000/", flush=True)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
