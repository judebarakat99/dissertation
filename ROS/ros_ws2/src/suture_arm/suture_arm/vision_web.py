#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import threading
from typing import Optional, List, Tuple

import numpy as np
import cv2
from flask import Flask, Response, render_template, abort, make_response, request, redirect, url_for
from ament_index_python.packages import get_package_share_directory

# ---- ZMQ Remote API client (fixed to port 23000) ----
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy

# ---- Torch / torchvision (for detection snapshot) ----
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

# ---------------- Configuration ----------------
CSIM_HOST = "127.0.0.1"
CSIM_PORT = 23000

TOP_SENSOR_ALIAS  = "/visionSensor"
SIDE_SENSOR_ALIAS = "/visionSensor_SideView"

FPS = 15
JPEG_QUALITY = 90

# Snapshot capture (right panel) uses stepped mode for stability.
STEPPED_SNAPSHOT = 1
# RAW streams (left column) use non-stepped readers like the older working version.
STEPPED_RAW = 0
# ------------------------------------------------


# ---------------- Template & models directories ----------------
def _find_templates_dir() -> str:
    env = os.getenv("SUTURE_ARM_TEMPLATES", "")
    if env and os.path.isdir(env):
        return env
    try:
        share = get_package_share_directory("suture_arm")
        tdir = os.path.join(share, "templates")
        if os.path.isdir(tdir):
            return tdir
    except Exception:
        pass
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "templates"))


def _find_models_root() -> str:
    env = os.getenv("SUTURE_ARM_ML", "")
    if env and os.path.isdir(env):
        return env
    try:
        share = get_package_share_directory("suture_arm")
        mdir = os.path.join(share, "ml")
        if os.path.isdir(mdir):
            return mdir
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


# ---------------- Snapshot vision capture (background thread) ----------------
class FrameGrabber:
    """Background grabber for TOP sensor frames used by the snapshot detection."""
    def __init__(self, host: str, port: int, sensor_alias: str):
        self.host = host
        self.port = port
        self.sensor_alias = sensor_alias
        self.client = None
        self.sim = None
        self.sensor = None
        self._last = None  # (bgr, ts)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _resolve(self, alias: str) -> Optional[int]:
        for cand in (alias, alias.lstrip('/'), alias + '#0'):
            try:
                return self.sim.getObject(cand)
            except Exception:
                pass
        return None

    def _connect(self):
        self.client = RemoteAPIClient(self.host, self.port)
        self.sim = self.client.require('sim')
        self.sensor = self._resolve(self.sensor_alias)
        if self.sensor is None:
            raise RuntimeError(f"Vision sensor '{self.sensor_alias}' not found")

        # Start if needed
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        # Stepping for stable snapshot grabs
        if STEPPED_SNAPSHOT and hasattr(self.sim, "setStepping"):
            try:
                self.sim.setStepping(True)
            except Exception:
                pass

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
            try:
                self.sim.step()
            except Exception:
                pass

        # Robust fetch across API variants
        try:
            img, res = self.sim.getVisionSensorImg(self.sensor)  # bytes, [w,h]
            w, h = int(res[0]), int(res[1])
        except Exception:
            try:
                img, w, h = self.sim.getVisionSensorCharImage(self.sensor)
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


# ---------------- Legacy RAW readers for top & side streams ----------------
_raw_ctx = {}  # alias -> {"client":..., "sim":..., "sensor":...}

def _raw_connect(alias: str):
    if alias in _raw_ctx and _raw_ctx[alias].get("sim") and _raw_ctx[alias].get("sensor"):
        return
    client = RemoteAPIClient(CSIM_HOST, CSIM_PORT)
    sim = client.require('sim')

    sensor = None
    for cand in (alias, alias.lstrip('/'), alias + '#0'):
        try:
            sensor = sim.getObject(cand)
            break
        except Exception:
            pass
    if sensor is None:
        raise RuntimeError(f"RAW: vision sensor '{alias}' not found")

    # Ensure simulation is running, but do NOT enable stepping here
    try:
        st = sim.getSimulationState()
        if st in (sim.simulation_stopped, sim.simulation_paused):
            sim.startSimulation()
    except Exception:
        pass

    _raw_ctx[alias] = {"client": client, "sim": sim, "sensor": sensor}


def _raw_read_once(alias: str) -> np.ndarray:
    ctx = _raw_ctx[alias]
    sim = ctx["sim"]; sensor = ctx["sensor"]

    try:
        if bool(sim.getObjectInt32Param(sensor, sim.visionintparam_explicit_handling)):
            sim.handleVisionSensor(sensor)
    except Exception:
        pass

    # Prefer the CharImage API first (what worked before)
    try:
        img, w, h = sim.getVisionSensorCharImage(sensor)
    except Exception:
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

    buf = np.frombuffer(img, dtype=np.uint8) if isinstance(img, (bytes, bytearray)) else np.array(img, dtype=np.uint8)
    n = buf.size

    if n % 3 == 0:
        frame = np.flip(buf.reshape(h, w, 3), 0)[:, :, ::-1]
    elif n == w * h:
        frame = cv2.cvtColor(np.flip(buf.reshape(h, w), 0), cv2.COLOR_GRAY2BGR)
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
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def mjpeg_generator_raw(alias: str):
    # Connect once per alias, read each frame freshly
    try:
        _raw_connect(alias)
    except Exception as e:
        msg = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(msg, f"RAW connect error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _encode_jpeg(msg) + b"\r\n")
        return

    period = 1.0 / max(1, FPS)
    while True:
        try:
            frame = _raw_read_once(alias)
            jpg = _encode_jpeg(frame)
        except Exception as e:
            err = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(err, f"RAW error: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            jpg = _encode_jpeg(err)
            time.sleep(0.1)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(period)


# ---------------- Detection: load + run (snapshot) ----------------
def _try_import_training_model(weights_path: str):
    """
    Try to import a model-builder that matches the weights file.
    Priority:
      1) <stem>.py sitting next to <stem>.pth   (e.g., mask_cuts_detector3.[pth|py])
      2) train_cuts_detector.py                 (legacy)
    Returns a callable (factory) or an already-instantiated model.
    """
    import importlib.util

    wdir = os.path.dirname(weights_path)
    stem = os.path.splitext(os.path.basename(weights_path))[0]

    candidate_modules = [
        os.path.join(wdir, f"{stem}.py"),
        os.path.join(wdir, "train_cuts_detector.py"),
    ]

    factories = (
        "get_model", "create_model", "build_model", "make_model",
        "get_detector", "create_detector",
    )
    classes = ("Model", "Detector")

    for module_path in candidate_modules:
        if not os.path.isfile(module_path):
            continue
        spec = importlib.util.spec_from_file_location(f"mdl_{stem}", module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        # 1) explicit factory names
        for name in factories:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        # 2) common class names that construct without args
        for cname in classes:
            cls = getattr(mod, cname, None)
            if cls is not None:
                try:
                    return cls  # will be called like a factory below
                except Exception:
                    pass
        # 3) direct model object
        mdl = getattr(mod, "model", None)
        if mdl is not None:
            return lambda **_: mdl
    return None


def _pick_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return None


def _build_fallback_model(num_classes: int = 2):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone="IMAGENET1K_V1",
        num_classes=num_classes
    )


def _load_detector(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location="cpu")

    class_names = None
    if isinstance(ckpt, dict):
        for key in ("classes", "class_names", "names", "labels"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                class_names = list(ckpt[key])
                break
        if class_names is None:
            # support dict mapping {class_name: idx}
            cti = ckpt.get("class_to_idx")
            if isinstance(cti, dict) and len(cti) > 0:
                # sort by index to get a stable list
                inv = sorted(cti.items(), key=lambda kv: int(kv[1]))
                class_names = [k for k, _ in inv]

    ctor = _try_import_training_model(weights_path)
    model = None
    if ctor:
        try:
            if class_names:
                if class_names and class_names[0].lower() in ("__background__", "background", "bg"):
                    num_classes = len(class_names)
                else:
                    num_classes = len(class_names) + 1
            else:
                num_classes = 2
            try:
                model = ctor(num_classes=num_classes)
            except TypeError:
                model = ctor()
        except Exception as e:
            print(f"[vision_web][WARN] train ctor failed: {e}")

    if model is None:
        if class_names:
            if class_names and class_names[0].lower() in ("__background__", "background", "bg"):
                num_classes = len(class_names)
            else:
                num_classes = len(class_names) + 1
        else:
            class_names = ["__background__", "cut"]
            num_classes = 2
        model = _build_fallback_model(num_classes=num_classes)

    # Accept a variety of checkpoint layouts
    state = (
        ckpt.get("model", None)
        or ckpt.get("model_state_dict", None)
        or ckpt.get("state_dict", None)
        or ckpt
    )
    try:
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print(f"[vision_web][WARN] non-strict load: {e}")
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    return model, class_names, device


def _draw_detections_pil(
    img_pil: "Image.Image",
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    class_names: List[str],
) -> "Image.Image":
    draw = ImageDraw.Draw(img_pil)
    font = _pick_font(16)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        color = tuple(int((hash(int(label)) >> (i * 8)) & 255) for i in range(3))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        lid = int(label)
        cls_name = class_names[lid] if 0 <= lid < len(class_names) else f"id_{lid}"
        text = f"{cls_name} {float(score):.2f}"
        if font:
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        else:
            tw, th = (8 * len(text), 14)
        pad = 2
        draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color)
        draw.text((x1 + pad, y1 - th - pad), text, fill=(255, 255, 255), font=font)
    return img_pil


@torch.inference_mode()
def _run_inference_on_frame(
    model, device, frame_bgr: np.ndarray,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5
) -> Tuple["Image.Image", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tens = F.to_tensor(img_pil).to(device)
    outputs = model([tens])[0]
    boxes = outputs.get("boxes", torch.empty((0, 4), device=device))
    scores = outputs.get("scores", torch.empty((0,), device=device))
    labels = outputs.get("labels", torch.empty((0,), dtype=torch.long, device=device))

    keep = scores >= conf_thresh
    boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]

    if boxes.numel() > 0:
        keep_idx = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep_idx]; scores = scores[keep_idx]; labels = labels[keep_idx]
    return img_pil, boxes.cpu(), scores.cpu(), labels.cpu()


def _encode_pil_jpeg(img_pil: "Image.Image", quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


# ---------------- Model selection & load ----------------
def _available_models() -> List[str]:
    try:
        files = [f for f in os.listdir(MODELS_ROOT) if f.lower().endswith(".pth")]
        files.sort()
        return files
    except Exception:
        return []


def _select_weights_file() -> Optional[str]:
    forced = os.getenv("SELECTED_WEIGHTS", "").strip()
    if forced:
        cand = forced if os.path.isabs(forced) else os.path.join(MODELS_ROOT, forced)
        if os.path.isfile(cand):
            return cand
    models = _available_models()
    return os.path.join(MODELS_ROOT, models[0]) if models else None


DETECTOR = None         # (model, class_names, device)
WEIGHTS_PATH = None     # absolute path used


# ---------------- Flask routes ----------------
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
        models=_available_models(),  # for dropdown
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

    global DETECTOR, WEIGHTS_PATH
    try:
        DETECTOR = _load_detector(candidate)
        WEIGHTS_PATH = candidate
        print(f"[vision_web] switched to model: {WEIGHTS_PATH}", flush=True)
    except Exception as e:
        print(f"[vision_web][ERR] failed to load {candidate}: {e}", flush=True)
    return redirect(url_for("index"))


@app.route("/stream_raw/<which>")
def stream_raw(which: str):
    alias = TOP_SENSOR_ALIAS if which == "top" else SIDE_SENSOR_ALIAS
    return Response(mjpeg_generator_raw(alias),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot_image():
    """One-shot detection overlay image (right panel), using the TOP view."""
    frame = grabber.get_frame()
    if frame is None:
        abort(503, "no frame yet")

    if not TORCH_OK or DETECTOR is None:
        # If no model, just return the raw frame to keep page useful
        jpg = _encode_jpeg(frame)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    model, class_names, device = DETECTOR
    try:
        img_pil, boxes, scores, labels = _run_inference_on_frame(model, device, frame)
        vis = _draw_detections_pil(img_pil, boxes, scores, labels,
                                   class_names or ["__background__", "cut"])
        jpg = _encode_pil_jpeg(vis, JPEG_QUALITY)
    except Exception as e:
        dbg = frame.copy()
        cv2.putText(dbg, f"inference error: {e}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        jpg = _encode_jpeg(dbg)

    resp = make_response(jpg)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/health")
def health():
    ok = grabber.get_frame() is not None
    torch_ok = TORCH_OK
    model_loaded = DETECTOR is not None
    return {
        "frame": ok,
        "torch": torch_ok,
        "model_loaded": model_loaded,
        "model_name": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
    }, 200 if ok else 503


# ---------------- Main ----------------
def main():
    print(f"[vision_web] templates dir: {TEMPLATES_DIR}", flush=True)
    print(f"[vision_web] models root  : {MODELS_ROOT}", flush=True)

    # start snapshot (TOP) capture thread
    grabber.start()
    t0 = time.time()
    while grabber.get_frame() is None and (time.time() - t0) < 5.0:
        time.sleep(0.05)

    # load a detector (once) if torch is available
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
