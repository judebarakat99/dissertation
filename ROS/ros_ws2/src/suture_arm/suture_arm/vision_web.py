#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import threading
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2
from flask import Flask, Response, render_template, abort, make_response, request, redirect, url_for, jsonify
from ament_index_python.packages import get_package_share_directory

# ---- ZMQ Remote API client ----
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy

# ---- Torch / torchvision ----
try:
    import torch
    import torchvision
    from torchvision.ops import nms
    from torchvision.transforms import functional as F
    from PIL import Image, ImageDraw, ImageFont
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    _TORCH_ERR = str(e)

# ---- Stitching ----
try:
    from . import stitching
except Exception:
    import stitching


# ====================== Configuration ======================

CSIM_HOST = "127.0.0.1"
CSIM_PORT = 23000

TOP_SENSOR_ALIAS = "/visionSensor"
SIDE_SENSOR_ALIAS = "/visionSensor_SideView"

FPS = 15
JPEG_QUALITY = 90

STEPPED_SNAPSHOT = 1
STEPPED_RAW = 0


# ====================== Templates & models ======================

def _find_templates_dir() -> str:
    env = os.getenv("SUTURE_ARM_TEMPLATES", "")
    if env and os.path.isdir(env): return env
    try:
        share = get_package_share_directory("suture_arm")
        tdir = os.path.join(share, "templates")
        if os.path.isdir(tdir): return tdir
    except Exception:
        pass
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "templates"))


def _find_models_root() -> str:
    env = os.getenv("SUTURE_ARM_ML", "")
    if env and os.path.isdir(env): return env
    try:
        share = get_package_share_directory("suture_arm")
        mdir = os.path.join(share, "ml")
        if os.path.isdir(mdir): return mdir
    except Exception:
        pass
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "ML_detection"))


TEMPLATES_DIR = _find_templates_dir()
MODELS_ROOT = _find_models_root()

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    raise RuntimeError(
        f"index.html not found in {TEMPLATES_DIR}. "
        "Place it under src/suture_arm/templates/ or set SUTURE_ARM_TEMPLATES."
    )

app = Flask(__name__, template_folder=TEMPLATES_DIR)


# ====================== Snapshot capture thread ======================

class FrameGrabber:
    def __init__(self, host: str, port: int, sensor_alias: str):
        self.host = host; self.port = port; self.sensor_alias = sensor_alias
        self.client = None; self.sim = None; self.sensor = None
        self._last = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _resolve(self, alias: str) -> Optional[int]:
        for cand in (alias, alias.lstrip("/"), alias + "#0"):
            try: return self.sim.getObject(cand)
            except Exception: pass
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
            try: self.sim.setStepping(True)
            except Exception: pass

    def _decode(self, img, w, h) -> np.ndarray:
        buf = np.frombuffer(img, dtype=np.uint8) if isinstance(img, (bytes, bytearray)) else np.array(img, dtype=np.uint8)
        n = buf.size
        if n == w*h:
            frame = cv2.cvtColor(np.flip(buf.reshape(h, w), 0), cv2.COLOR_GRAY2BGR)
        elif n == w*h*3:
            frame = np.flip(buf.reshape(h, w, 3), 0)[:, :, ::-1]
        elif n == w*h*4:
            frame = np.flip(buf.reshape(h, w, 4), 0)[:, :, :3][:, :, ::-1]
        else:
            if (w*h) and n % (w*h) == 0:
                c = n // (w*h)
                frame = np.flip(buf.reshape(h, w, c), 0)[:, :, :3][:, :, ::-1]
            else:
                raise RuntimeError(f"unexpected buffer size n={n} vs {w}x{h}")
        if not (frame.flags["C_CONTIGUOUS"] and frame.flags["WRITEABLE"]):
            frame = np.ascontiguousarray(frame.copy())
        return frame

    def _read_frame(self) -> np.ndarray:
        try:
            if bool(self.sim.getObjectInt32Param(self.sensor, self.sim.visionintparam_explicit_handling)):
                self.sim.handleVisionSensor(self.sensor)
        except Exception:
            pass
        if STEPPED_SNAPSHOT and hasattr(self.sim, "setStepping"):
            try: self.sim.step()
            except Exception: pass
        try:
            img, res = self.sim.getVisionSensorImg(self.sensor)  # bytes, [w,h]
            w, h = int(res[0]), int(res[1])
        except Exception as e1:
            try:
                out = self.sim.getVisionSensorImage(self.sensor)
                if isinstance(out, (list, tuple)) and len(out) == 3 and isinstance(out[1], (int, float)):
                    img, w, h = out; w, h = int(w), int(h)
                else:
                    img, res = out; w, h = int(res[0]), int(res[1])
                if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
                    arr = np.array(img, dtype=np.float32)
                    img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
            except Exception as e2:
                raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")
        return self._decode(img, w, h)

    def _loop(self):
        period = 1.0 / max(1, FPS)
        while not self._stop.is_set():
            try:
                if self.sim is None: self._connect()
                frame = self._read_frame()
                with self._lock: self._last = (frame, time.time())
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
            if self._last is None: return None
            return self._last[0].copy()


grabber = FrameGrabber(CSIM_HOST, CSIM_PORT, TOP_SENSOR_ALIAS)


# ====================== RAW readers (Top/Side) ======================

_raw_ctx: Dict[str, Dict] = {}

def _raw_connect(alias: str):
    if alias in _raw_ctx and _raw_ctx[alias].get("sim") and _raw_ctx[alias].get("sensor"): return
    client = RemoteAPIClient(CSIM_HOST, CSIM_PORT)
    sim = client.require("sim")
    sensor = None
    for cand in (alias, alias.lstrip("/"), alias + "#0"):
        try: sensor = sim.getObject(cand); break
        except Exception: pass
    if sensor is None:
        raise RuntimeError(f"RAW: vision sensor '{alias}' not found")
    try:
        st = sim.getSimulationState()
        if st in (sim.simulation_stopped, sim.simulation_paused): sim.startSimulation()
    except Exception:
        pass
    _raw_ctx[alias] = {"client": client, "sim": sim, "sensor": sensor}


def _raw_read_once(alias: str) -> np.ndarray:
    ctx = _raw_ctx[alias]; sim = ctx["sim"]; sensor = ctx["sensor"]
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
            img, w, h = out; w, h = int(w), int(h)
        else:
            img, res = out; w, h = int(res[0]), int(res[1])
        if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
            arr = np.array(img, dtype=np.float32)
            img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
    buf = np.frombuffer(img, dtype=np.uint8); n = buf.size
    if n == w*h:
        frame = cv2.cvtColor(np.flip(buf.reshape(h, w), 0), cv2.COLOR_GRAY2BGR)
    elif n == w*h*3:
        frame = np.flip(buf.reshape(h, w, 3), 0)[:, :, ::-1]
    elif n == w*h*4:
        frame = np.flip(buf.reshape(h, w, 4), 0)[:, :, :3][:, :, ::-1]
    else:
        if (w*h) and n % (w*h) == 0:
            c = n // (w*h)
            frame = np.flip(buf.reshape(h, w, c), 0)[:, :, :3][:, :, ::-1]
        else:
            raise RuntimeError(f"RAW({alias}): unexpected buffer size n={n} vs {w}x{h}")
    if not (frame.flags["C_CONTIGUOUS"] and frame.flags["WRITEABLE"]):
        frame = np.ascontiguousarray(frame.copy())
    return frame


def _encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def mjpeg_generator_raw(alias: str):
    try:
        _raw_connect(alias)
    except Exception as e:
        msg = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(msg, f"RAW connect error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _encode_jpeg(msg) + b"\r\n"); return
    period = 1.0 / max(1, FPS)
    while True:
        try:
            frame = _raw_read_once(alias); jpg = _encode_jpeg(frame)
        except Exception as e:
            err = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(err, f"RAW error: {str(e)[:40]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            jpg = _encode_jpeg(err); time.sleep(0.1)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(period)


# ====================== Detection & masks ======================

def _try_import_training_model(weights_path: str):
    import importlib.util
    wdir = os.path.dirname(weights_path)
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    candidates = [
        os.path.join(wdir, f"{stem}.py"),
        os.path.join(wdir, "train_cuts_detector.py"),
    ]
    factories = ("get_model", "create_model", "build_model", "make_model", "get_detector", "create_detector")
    classes = ("Model", "Detector")
    for module_path in candidates:
        if not os.path.isfile(module_path): continue
        spec = importlib.util.spec_from_file_location(f"mdl_{stem}", module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        for name in factories:
            fn = getattr(mod, name, None)
            if callable(fn): return fn
        for cname in classes:
            cls = getattr(mod, cname, None)
            if cls is not None: return cls
        mdl = getattr(mod, "model", None)
        if mdl is not None: return lambda **_: mdl
    return None


def _pick_font(size: int = 16):
    try: return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception: return None


def _guess_wants_masks(weights_path: str, ckpt: dict) -> bool:
    name = os.path.basename(weights_path).lower()
    if "mask" in name or "maskrcnn" in name:
        return True
    state = ckpt.get("model", None) or ckpt.get("state_dict", None) or ckpt
    try:
        for k in state.keys():
            if "mask" in k.lower(): return True
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


def _load_detector(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location="cpu")

    class_names = None
    if isinstance(ckpt, dict):
        for key in ("classes", "class_names", "names", "labels"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                class_names = list(ckpt[key]); break
        if class_names is None:
            cti = ckpt.get("class_to_idx")
            if isinstance(cti, dict) and len(cti) > 0:
                inv = sorted(cti.items(), key=lambda kv: int(kv[1]))
                class_names = [k for k, _ in inv]

    ctor = _try_import_training_model(weights_path)
    model = None
    if ctor:
        try:
            if class_names:
                num_classes = len(class_names) if class_names[0].lower() in ("__background__", "background", "bg") else len(class_names) + 1
            else:
                num_classes = 2
            try: model = ctor(num_classes=num_classes)
            except TypeError: model = ctor()
        except Exception as e:
            print(f"[vision_web][WARN] train ctor failed: {e}")

    if model is None:
        if class_names:
            num_classes = len(class_names) if class_names[0].lower() in ("__background__", "background", "bg") else len(class_names) + 1
        else:
            class_names = ["__background__", "cut"]; num_classes = 2
        want_masks = _guess_wants_masks(weights_path, ckpt if isinstance(ckpt, dict) else {})
        model = _build_fallback_model(num_classes=num_classes, want_masks=want_masks)

    state = ckpt.get("model", None) or ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None) or ckpt
    try:
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print(f"[vision_web][WARN] non-strict load: {e}")
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    has_mask = any("mask" in n.lower() for n, _ in model.named_modules())
    print(f"[vision_web] model loaded. mask_head={has_mask}", flush=True)
    return model, class_names, device


# -------- Robust mask extraction --------

MASK_BIN_THR = 0.5  # threshold after sigmoid if logits are provided

def _to_numpy_uint8_mask(m: "torch.Tensor") -> np.ndarray:
    import torch
    if isinstance(m, (list, tuple)):
        m = torch.stack([torch.as_tensor(x) for x in m], dim=0)
    t = torch.as_tensor(m)
    if t.dtype == torch.bool:
        t = t.to(dtype=torch.uint8) * 255
        if t.dim() == 2: t = t.unsqueeze(0)
        if t.dim() == 4 and t.shape[1] == 1: t = t[:, 0]
        return t.detach().cpu().numpy()
    if t.dtype.is_floating_point and (t.min() < 0 or t.max() > 1):
        t = t.sigmoid()
    if t.dim() == 2: t = t.unsqueeze(0)
    elif t.dim() == 4:
        if t.shape[1] == 1: t = t[:, 0]
        else: t, _ = t.max(dim=1)
    t = (t >= MASK_BIN_THR).to(dtype=torch.uint8) * 255
    return t.detach().cpu().numpy()


@torch.inference_mode()
def _run_inference_on_frame(model, device, frame_bgr: np.ndarray, conf_thresh: float = 0.4, iou_thresh: float = 0.5):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tens = F.to_tensor(img_pil).to(device)

    out = model([tens])
    outputs = out[0] if isinstance(out, (list, tuple)) else out

    is_dict = isinstance(outputs, dict)
    boxes  = outputs.get("boxes",  None) if is_dict else None
    scores = outputs.get("scores", None) if is_dict else None
    labels = outputs.get("labels", None) if is_dict else None

    masks = None
    if is_dict:
        for k in ("masks", "pred_masks", "mask", "segmentation", "segm", "probs_mask"):
            if k in outputs: masks = outputs[k]; break
    try:
        if masks is None and hasattr(outputs, "get_fields"):
            fields = outputs.get_fields()
            if "pred_masks" in fields: masks = fields["pred_masks"]
            boxes  = boxes  if boxes  is not None else fields.get("pred_boxes", None)
            scores = scores if scores is not None else fields.get("scores", None)
            labels = labels if labels is not None else fields.get("pred_classes", None)
    except Exception:
        pass

    def _to_tensor(x, default_shape=()):
        try: return torch.as_tensor(x)
        except Exception:
            try: return torch.as_tensor(x.tensor)
            except Exception: return torch.empty(default_shape)

    boxes  = _to_tensor(boxes, (0, 4))
    scores = _to_tensor(scores, (0,))
    labels = _to_tensor(labels, (0,)).to(dtype=torch.long)

    keep = (scores >= conf_thresh)
    if keep.numel() > 0:
        boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
        if masks is not None:
            try: masks = masks[keep]
            except Exception:
                if isinstance(masks, (list, tuple)):
                    masks = [m for m, k in zip(masks, keep.tolist()) if k]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes  = boxes[keep_idx]; scores = scores[keep_idx]; labels = labels[keep_idx]
        if masks is not None:
            try: masks = masks[keep_idx]
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


def _parse_hex_color(s: Optional[str], default=(0, 220, 220)) -> Tuple[int, int, int]:
    if not s: return default
    try:
        s = s.strip()
        if s.startswith('#'): s = s[1:]
        if len(s) == 3: s = ''.join([ch * 2 for ch in s])
        if len(s) != 6: return default
        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return default


def _draw_detections_pil(img_pil, boxes, scores, labels, class_names, selected_index=None, masks=None,
                         poly_color=(0,220,220), poly_width=3):
    draw = ImageDraw.Draw(img_pil)
    font = _pick_font(16)

    # Outline mask polygons
    if masks is not None:
        is_torch = False
        try:
            import torch as _t
            is_torch = isinstance(masks, _t.Tensor)
        except Exception:
            pass
        if is_torch:
            M = masks.shape[0]
            def get_mask(k): return masks[k,0].detach().cpu().numpy() if masks.ndim==4 else masks[k].detach().cpu().numpy()
        else:
            M = masks.shape[0]
            def get_mask(k): return masks[k]
        for idx in range(M):
            m = get_mask(idx)
            mb = (m >= 128).astype(np.uint8)
            cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (255,200,30) if (selected_index is not None and idx == selected_index) else poly_color
            for c in cnts:
                if len(c) < 3: continue
                poly = [(int(p[0][0]), int(p[0][1])) for p in c]
                draw.line(poly + [poly[0]], fill=color, width=int(poly_width))

    # Draw widened boxes + ids
    W, H = img_pil.size
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        w = x2 - x1; h = y2 - y1
        pad_w_vis = 0.22 * w * 2.0  # widen horizontally (~44% total extra width)
        pad_h_vis = 0.08 * h        # modest vertical pad
        vx1 = max(0.0, x1 - pad_w_vis)
        vy1 = max(0.0, y1 - pad_h_vis)
        vx2 = min(float(W), x2 + pad_w_vis)
        vy2 = min(float(H), y2 + pad_h_vis)

        color = (80,180,255) if (selected_index is not None and idx == selected_index) \
                else tuple(int((hash(int(label)) >> (i*8)) & 255) for i in range(3))
        width = 5 if (selected_index is not None and idx == selected_index) else 3
        draw.rectangle([vx1, vy1, vx2, vy2], outline=color, width=width)

        lid = int(label)
        cls_name = class_names[lid] if 0 <= lid < len(class_names) else f"id_{lid}"
        text = f"{cls_name} {float(score):.2f}"
        if font: tw, th = draw.textbbox((0,0), text, font=font)[2:]
        else:    tw, th = (8*len(text), 14)
        pad = 2
        draw.rectangle([vx1, vy1 - th - 2*pad, vx1 + tw + 2*pad, vy1], fill=color)
        draw.text((vx1 + pad, vy1 - th - pad), text, fill=(255,255,255), font=font)
        idx_text = f"#{idx}"
        if font: itw, ith = draw.textbbox((0,0), idx_text, font=font)[2:]
        else:    itw, ith = (12,12)
        tag_pad = 2
        draw.rectangle([vx1, vy1, vx1 + itw + 2*tag_pad, vy1 + ith + 2*tag_pad], fill=(0,0,0))
        draw.text((vx1 + tag_pad, vy1 + tag_pad), idx_text, fill=(255,255,255), font=font)
    return img_pil


def _encode_pil_jpeg(img_pil, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


# ====================== Model management ======================

def _available_models() -> List[str]:
    try:
        files = [f for f in os.listdir(MODELS_ROOT) if f.lower().endswith(".pth")]
        files.sort(key=lambda n: (0 if "cuts_maskrcnn_best" in n.lower() else 1, n.lower()))
        return files
    except Exception:
        return []


def _select_weights_file() -> Optional[str]:
    models = _available_models()
    if not models: return None
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
    if not fname: return redirect(url_for("index"))
    candidate = os.path.join(MODELS_ROOT, fname)
    if not os.path.isfile(candidate): return redirect(url_for("index"))
    if not TORCH_OK: return redirect(url_for("index"))

    global DETECTOR, WEIGHTS_PATH, LAST_DETS, SELECTED_DET_ID, LAST_FRAME_PIL, LAST_MASKS
    try:
        DETECTOR = _load_detector(candidate)
        WEIGHTS_PATH = candidate
        LAST_DETS = None; LAST_FRAME_PIL = None; LAST_MASKS = None
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
    if frame is None: abort(503, "no frame yet")

    if not TORCH_OK or DETECTOR is None:
        jpg = _encode_jpeg(frame)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    poly_color = _parse_hex_color(request.args.get("poly_color"), default=(0, 220, 220))
    poly_width = request.args.get("poly_width", default=3, type=int)

    model, class_names, device = DETECTOR
    try:
        img_pil, boxes, scores, labels, masks_np = _run_inference_on_frame(model, device, frame)

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

        vis = _draw_detections_pil(
            img_pil, boxes, scores, labels, cnames,
            selected_index=SELECTED_DET_ID, masks=masks_np,
            poly_color=poly_color, poly_width=int(poly_width)
        )
        jpg = _encode_pil_jpeg(vis, JPEG_QUALITY)
    except Exception as e:
        dbg = frame.copy()
        cv2.putText(dbg, f"inference error: {e}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        jpg = _encode_jpeg(dbg)

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
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        cx = x1 + 0.5*w; cy = y1 + 0.5*h
        lab_name = cnames[int(l)] if (0 <= int(l) < len(cnames)) else f"id_{int(l)}"
        dets.append({"id": i, "score": float(s), "label": int(l), "label_name": lab_name,
                     "box": [x1,y1,x2,y2], "cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})
    return jsonify({"detections": dets, "selected_id": SELECTED_DET_ID,
                    "model": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
                    "ts": LAST_DETS.get("ts"), "size": LAST_DETS.get("image_size")})


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
    """Return (crop_bgr, crop_mask) using **asymmetric padding** (wider horizontally).
       You can override via query: &pad_w=..&pad_h=..
    """
    if LAST_DETS is None or LAST_FRAME_PIL is None:
        raise RuntimeError("No snapshot available. Click Snapshot first.")
    if det_id is None: det_id = SELECTED_DET_ID
    if det_id is None: raise RuntimeError("No detection selected.")
    boxes = LAST_DETS.get("boxes", [])
    if det_id < 0 or det_id >= len(boxes): raise RuntimeError(f"Selected id {det_id} out of range.")
    if LAST_MASKS is None or det_id >= len(LAST_MASKS) or LAST_MASKS[det_id] is None:
        raise RuntimeError("ML mask not available. Load cuts_maskrcnn_best.pth and refresh Snapshot.")

    x1, y1, x2, y2 = [float(v) for v in boxes[det_id]]
    W, H = LAST_DETS.get("image_size", [LAST_FRAME_PIL.width, LAST_FRAME_PIL.height])

    # Wider horizontally by default
    pad_w = float(request.args.get("pad_w", default=pad * 2.2))
    pad_h = float(request.args.get("pad_h", default=pad))
    pw = pad_w * (x2 - x1); ph = pad_h * (y2 - y1)

    xx1 = max(0, int(np.floor(x1 - pw)))-10; yy1 = max(0, int(np.floor(y1 - ph)))
    xx2 = 10+min(int(np.ceil(x2 + pw)), int(W)); yy2 = min(int(np.ceil(y2 + ph)), int(H))

    crop_pil = LAST_FRAME_PIL.crop((xx1, yy1, xx2, yy2))
    crop_rgb = np.array(crop_pil)
    if crop_rgb.ndim == 2: crop_rgb = np.repeat(crop_rgb[..., None], 3, axis=2)
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
        jpg = _encode_jpeg(crop_bgr)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((120, 480, 3), np.uint8)
        cv2.putText(err, f"crop err: {e}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        jpg = _encode_jpeg(err)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


@app.route("/mask.jpg")
def mask_image():
    try:
        det_id = request.args.get("id", None, type=int)
        pad = request.args.get("pad", default=0.08, type=float)
        outw = request.args.get("size", default=None, type=int)
        crop_bgr, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        vis = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
        jpg = _encode_jpeg(vis, JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((160, 680, 3), np.uint8)
        cv2.putText(err, f"mask error: {e}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        jpg = _encode_jpeg(err)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


# --- in vision_web.py ---
@app.route("/pattern.jpg")
def pattern_image():
    try:
        # ---- required / common args ----
        det_id = request.args.get("id", None, type=int)
        pad    = request.args.get("pad",  default=0.08, type=float)
        outw   = request.args.get("size", default=None, type=int)

        # pattern mode: perp | continuous | mold | bezier
        style  = request.args.get("pattern", default="continuous", type=str).lower()

        # spacing & legacy args (bite/s_min/s_max kept for compat; not used here)
        spacing = request.args.get("spacing", default=20, type=int)
        bite    = request.args.get("bite",    default=0.9, type=float)
        _smin   = request.args.get("s_min",   default=8,   type=int)
        _smax   = request.args.get("s_max",   default=60,  type=int)

        # robustness / tuning knobs
        curvature_gain = request.args.get("curvature_gain", default=18.0, type=float)
        alpha_alias    = request.args.get("alpha", default=None, type=float)  # UI may send 'alpha'
        if alpha_alias is not None:
            curvature_gain = float(alpha_alias)

        outside_scale  = request.args.get("outside_scale",  default=0.12, type=float)
        spur_min_px    = request.args.get("spur_min_px",    default=4,    type=int)
        border_push_px = request.args.get("border_push_px", default=0.0,  type=float)

        # Clinical knob: entry distance in mm (we clamp to 5–10 mm as requested)
        entry_mm  = request.args.get("entry_mm",  default=6.0, type=float)
        entry_mm  = float(min(10.0, max(5.0, entry_mm)))  # enforce 5–10 mm

        # Optional calibration: pixels per mm. If missing, try to auto-guess from mat size.
        px_per_mm = request.args.get("px_per_mm", default=None, type=float)

        # Bézier-only knob
        bez_samples    = request.args.get("bez_samples",    default=16,   type=int)

        # mode-specific extras (sane defaults)
        outside_px     = request.args.get("outside_px",     default=3.0,  type=float)
        rect_min_step  = request.args.get("rect_min_step",  default=6.0,  type=float)
        max_probe_arg  = request.args.get("max_probe",      default=None, type=int)

        # thread styling / color parsing
        def _parse_hex_color(s, default=(30, 200, 255)):
            if not s:
                return default
            s = s.strip()
            if s[0] == '#':
                s = s[1:]
            if len(s) == 3:
                s = ''.join([c*2 for c in s])
            if len(s) != 6:
                return default
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (b, g, r)  # OpenCV uses BGR

        thread_color = _parse_hex_color(request.args.get("thread_color"), default=(30, 200, 255))
        thread_thick = request.args.get("thread_thick", default=2, type=int)
        debug_flag   = request.args.get("debug", default=0, type=int) == 1

        # ---- fetch crop + mask from current detection ----
        crop_bgr, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)

        # safety for mask dtype/shape
        if crop_mask is None or crop_mask.size == 0:
            crop_mask = np.zeros(crop_bgr.shape[:2], np.uint8)
        elif crop_mask.ndim == 3:
            crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

        # ---- AUTO mm→px calibration (if user didn't supply px_per_mm) ----
        # The training mat is 120mm (width) x 175mm (height). If the crop likely
        # shows the whole mat, derive px/mm from crop size. Otherwise user-supplied
        # px_per_mm overrides this.
        if px_per_mm in (None, ""):
            H, W = crop_bgr.shape[:2]
            # Always compute both directions and average; user-specified value will override anyway.
            ppm_w = float(W) / 120.0   # px per mm assuming full width = 120 mm
            ppm_h = float(H) / 175.0   # px per mm assuming full height = 175 mm
            # Heuristic: ignore obviously tiny crops (avoid blowing up when zoomed in)
            # If the crop is small, this stays a rough guess but will still let the slider have an effect.
            px_per_mm = float(max(0.0, (ppm_w + ppm_h) * 0.5))

        # max_probe default: depend on current crop size
        max_probe = max_probe_arg if max_probe_arg is not None else int(max(crop_mask.shape) * 2.0)

        # ---- dispatch to the selected pattern renderer ----
        over = None

        common_kwargs = dict(
            spacing_px=float(spacing),
            max_probe=int(max_probe),
            color_thread=thread_color,
            thickness=int(thread_thick),
            debug=debug_flag,
            curvature_gain=float(curvature_gain),
            outside_scale=float(outside_scale),
            spur_min_px=int(spur_min_px),
            # pass clamped mm and our (possibly auto-guessed) px/mm
            entry_mm=float(entry_mm),
            px_per_mm=(float(px_per_mm) if px_per_mm not in (None, "") else None),
        )

        if style == "perp":
            over, _ = stitching.draw_stitching_pattern(
                crop_bgr, crop_mask, info={},
                spacing=int(spacing),
                length_scale=2.0,
                color=thread_color,
                thickness=int(thread_thick),
                curvature_gain=float(curvature_gain),
                outside_scale=float(outside_scale),
                spur_min_px=int(spur_min_px),
                entry_mm=float(entry_mm),
                px_per_mm=(float(px_per_mm) if px_per_mm not in (None, "") else None),
            )

        elif style == "continuous":
            over, _ = stitching.draw_running_suture_auto(
                crop_bgr, crop_mask,
                outside_px=float(outside_px),
                rect_min_step=float(rect_min_step),
                **common_kwargs,
            )

        elif style == "mold":
            over, _ = stitching.draw_stitching_mold_border(
                crop_bgr, crop_mask,
                grow_px=3,
                border_push_px=float(border_push_px),
                **common_kwargs,
            )

        elif style == "bezier":
            draw_fn = getattr(stitching, "draw_stitching_bezier", None)
            if draw_fn is None:
                over, _ = stitching.draw_running_suture_auto(
                    crop_bgr, crop_mask,
                    outside_px=float(outside_px),
                    rect_min_step=float(rect_min_step),
                    **common_kwargs,
                )
            else:
                over, _ = draw_fn(
                    crop_bgr, crop_mask,
                    outside_px=float(outside_px),
                    rect_min_step=float(rect_min_step),
                    bezier_samples_per=int(bez_samples),
                    **common_kwargs,
                )

        else:
            fn_name = f"draw_stitching_{style}"
            draw_fn = getattr(stitching, fn_name, None)
            if draw_fn is None:
                over, _ = stitching.draw_stitching_mold_border(
                    crop_bgr, crop_mask,
                    grow_px=3,
                    border_push_px=float(border_push_px),
                    **common_kwargs,
                )
            else:
                over, _ = draw_fn(
                    crop_bgr, crop_mask,
                    **common_kwargs,
                )

        # ---- encode & return ----
        jpg = _encode_jpeg(over, JPEG_QUALITY)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        err = np.zeros((220, 980, 3), np.uint8)
        cv2.putText(err, f"pattern error: {e}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        jpg = _encode_jpeg(err, JPEG_QUALITY)
        resp = make_response(jpg)
        resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


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

    grabber.start()
    t0 = time.time()
    while grabber.get_frame() is None and (time.time() - t0) < 5.0:
        time.sleep(0.05)

    global DETECTOR, WEIGHTS_PATH
    if TORCH_OK:
        WEIGHTS_PATH = _select_weights_file()
        if WEIGHTS_PATH and os.path.isfile(WEIGHTS_PATH):
            try:
                DETECTOR = _load_detector(WEIGHTS_PATH)
                print(f"[vision_web] loaded weights: {WEIGHTS_PATH}", flush=True)
            except Exception as e:
                print(f"[vision_web][ERR] failed to load detector: {e}", flush=True)
                DETECTOR = None
        else:
            print("[vision_web] no *.pth found under models root", flush=True)
    else:
        print(f"[vision_web][WARN] PyTorch unavailable: {_TORCH_ERR}", flush=True)

    print(f"[vision_web] open http://127.0.0.1:8000/", flush=True)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
