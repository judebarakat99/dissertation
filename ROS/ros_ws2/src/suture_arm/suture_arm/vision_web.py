#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import types
import threading
import importlib.util
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from flask import Flask, Response, render_template

# ----- ROS2 share discovery for templates -----
from ament_index_python.packages import get_package_share_directory

# ----- ZMQ Remote API client (fixed port 23000 as requested) -----
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy name, if needed

# ---------------- Configuration ----------------
CSIM_HOST = "127.0.0.1"
CSIM_PORT = 23000
SENSOR_ALIAS = "/visionSensor"     # change here if your alias differs
FPS = 15
JPEG_QUALITY = 90
STEPPED = 1                        # use stepped mode for stable reads

MODEL_STEMS = [
    #"mask_cuts_detector",
    "cuts_detector_best",
    # "mask_cuts_detector2",
    # "cuts_detector_best2",
    # "mask_cuts_detector3",
]
# ------------------------------------------------


def _find_templates_dir() -> str:
    # 0) optional override for debugging
    env_tdir = os.getenv("SUTURE_ARM_TEMPLATES", "")
    if env_tdir and os.path.isdir(env_tdir):
        return env_tdir
    # 1) installed share dir
    try:
        share = get_package_share_directory("suture_arm")
        tdir = os.path.join(share, "templates")
        if os.path.isdir(tdir):
            return tdir
    except Exception:
        pass
    # 2) fallback to source tree
    here = os.path.dirname(__file__)
    tdir = os.path.abspath(os.path.join(here, "..", "templates"))
    return tdir


def _find_models_root() -> str:
    # 0) optional override
    env_mdir = os.getenv("SUTURE_ARM_ML", "")
    if env_mdir and os.path.isdir(env_mdir):
        return env_mdir
    # 1) installed share dir
    try:
        share = get_package_share_directory("suture_arm")
        mdir = os.path.join(share, "ml")
        if os.path.isdir(mdir):
            return mdir
    except Exception:
        pass
    # 2) fallback to source tree
    here = os.path.dirname(__file__)
    mdir = os.path.abspath(os.path.join(here, "..", "ML_detection"))
    return mdir


TEMPLATES_DIR = _find_templates_dir()
MODELS_ROOT = _find_models_root()

print(f"[vision_web] templates dir: {TEMPLATES_DIR}", flush=True)
print(f"[vision_web] models root  : {MODELS_ROOT}", flush=True)

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    raise RuntimeError(
        f"index.html not found in {TEMPLATES_DIR}. "
        "Ensure it's installed to share/suture_arm/templates or set SUTURE_ARM_TEMPLATES."
    )

app = Flask(__name__, template_folder=TEMPLATES_DIR)


# --------------- Single capture thread ---------------
class FrameGrabber:
    def __init__(self, host: str, port: int, sensor_alias: str):
        self.host = host
        self.port = port
        self.sensor_alias = sensor_alias
        self.client = None
        self.sim = None
        self.sensor = None
        self._lock = threading.Lock()
        self._last = None  # (bgr, ts)
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

        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        if STEPPED and hasattr(self.sim, "setStepping"):
            try:
                self.sim.setStepping(True)
            except Exception:
                pass

    def _read_frame(self) -> np.ndarray:
        # Handle explicitHandling if enabled
        try:
            if bool(self.sim.getObjectInt32Param(self.sensor, self.sim.visionintparam_explicit_handling)):
                self.sim.handleVisionSensor(self.sensor)
        except Exception:
            pass

        if STEPPED and hasattr(self.sim, "setStepping"):
            try:
                self.sim.step()
            except Exception:
                pass

        # Robust fetch across API variants
        try:
            img, res = self.sim.getVisionSensorImg(self.sensor)  # (buf, [w,h])
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
                print(f"[vision_web][ERR] capture: {e}", flush=True)
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


grabber = FrameGrabber(CSIM_HOST, CSIM_PORT, SENSOR_ALIAS)


# --------------- Model wrappers (no sim calls here) ---------------
class ModelWrapper:
    def __init__(self, root: str, stem: str):
        self.root = root
        self.stem = stem
        self.pth = os.path.join(root, stem + ".pth")
        self.py  = os.path.join(root, stem + ".py")
        self.mod: Optional[types.ModuleType] = None
        self.handle = None
        self.ok = False
        self.err = None
        self._load()

    def _load(self):
        try:
            if not os.path.isfile(self.py):
                raise FileNotFoundError(f"wrapper not found: {self.py}")
            if not os.path.isfile(self.pth):
                raise FileNotFoundError(f"model not found: {self.pth}")

            spec = importlib.util.spec_from_file_location(f"suture_arm.plugins.{self.stem}", self.py)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            if not hasattr(mod, "load") or not hasattr(mod, "predict"):
                raise RuntimeError("wrapper must define load() and predict()")
            self.mod = mod
            self.handle = mod.load(self.pth)
            self.ok = True
        except Exception as e:
            self.err = str(e)
            self.ok = False

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        if not self.ok:
            raise RuntimeError(self.err or "model not loaded")
        return self.mod.predict(self.handle, frame_bgr)


def load_all_models() -> Dict[str, ModelWrapper]:
    models: Dict[str, ModelWrapper] = {}
    for stem in MODEL_STEMS:
        mw = ModelWrapper(MODELS_ROOT, stem)
        models[stem] = mw
        if mw.ok:
            print(f"[vision_web] loaded: {stem}", flush=True)
        else:
            print(f"[vision_web] WARN: {stem} not loaded -> {mw.err}", flush=True)
    return models


MODELS: Dict[str, ModelWrapper] = {}


# ---------------- Overlay renderer ----------------
def draw_overlay(frame_bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    img = frame_bgr.copy()
    try:
        cuts = result.get("cuts", [])
        for c in cuts:
            poly = c.get("polyline", [])
            pts = np.array(poly, dtype=float)
            # Here we assume the wrapper returns **pixel** polylines. If your
            # wrappers output meters, add projection in the wrapper so they
            # return pixels for the browser overlay.
            if pts.ndim == 2 and pts.shape[1] >= 2:
                pts = pts.astype(int)
                for i in range(len(pts) - 1):
                    cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]),
                             (40, 255, 40), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(img, f"overlay error: {e}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ---------------- MJPEG generators (no sim calls) ----------------
def _encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def mjpeg_generator_raw():
    while True:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.05); continue
        try:
            jpg = _encode_jpeg(frame)
        except Exception:
            time.sleep(0.02); continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1.0 / max(1, FPS))


def mjpeg_generator_model(stem: str):
    mw = MODELS.get(stem)
    if mw is None or not mw.ok:
        # persistent error card
        canvas = np.zeros((240, 320, 3), np.uint8)
        msg = f"{stem}: not loaded"
        cv2.putText(canvas, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        err_jpg = _encode_jpeg(canvas, 90)
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + err_jpg + b"\r\n")
            time.sleep(0.5)

    while True:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.05); continue
        try:
            result = mw.predict(frame)
            drawn = draw_overlay(frame, result)
            jpg = _encode_jpeg(drawn)
        except Exception as e:
            drawn = frame.copy()
            cv2.putText(drawn, f"{stem} error: {e}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            jpg = _encode_jpeg(drawn)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1.0 / max(1, FPS))


# ---------------- Routes ----------------
@app.route("/")
def index():
    loaded = [k for k, v in MODELS.items() if v.ok]
    return render_template("index.html",
                           sensor=SENSOR_ALIAS,
                           host=CSIM_HOST,
                           port=CSIM_PORT,
                           loaded_models=loaded)

@app.route("/stream_raw")
def stream_raw():
    return Response(mjpeg_generator_raw(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_model/<stem>")
def stream_model(stem: str):
    return Response(mjpeg_generator_model(stem),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- Main entry ----------------
def main():
    # start capture first
    grabber.start()

    # warm up a bit for first frame
    t0 = time.time()
    while grabber.get_frame() is None and (time.time() - t0) < 5.0:
        time.sleep(0.05)

    # load models (no sim calls)
    global MODELS
    MODELS = load_all_models()

    print(f"[vision_web] connecting   : {CSIM_HOST}:{CSIM_PORT}", flush=True)
    print(f"[vision_web] ready on     : http://0.0.0.0:8000/", flush=True)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
