#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
from typing import Optional, Tuple

import numpy as np
import cv2

# ---- ZMQ Remote API client ----
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    # legacy name in older CoppeliaSim installs
    from zmqRemoteApi import RemoteAPIClient


# ====================== Helpers ======================

def _resolve_sensor(sim, alias: str) -> Optional[int]:
    """Try common alias variations and return object handle or None."""
    for cand in (alias, alias.lstrip("/"), alias + "#0"):
        try:
            return sim.getObject(cand)
        except Exception:
            pass
    return None


def _decode_img(buf, w: int, h: int) -> np.ndarray:
    """Decode CoppeliaSim sensor buffer into BGR uint8 image."""
    arr = np.frombuffer(buf, dtype=np.uint8) if isinstance(buf, (bytes, bytearray)) else np.asarray(buf, dtype=np.uint8)
    n = arr.size

    if n == w * h:
        # grayscale
        img = np.flip(arr.reshape(h, w), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif n == w * h * 3:
        # RGB
        img = np.flip(arr.reshape(h, w, 3), 0)[:, :, ::-1]  # to BGR
    elif n == w * h * 4:
        # RGBA
        img = np.flip(arr.reshape(h, w, 4), 0)[:, :, :3][:, :, ::-1]
    else:
        if (w * h) and n % (w * h) == 0:
            c = n // (w * h)
            img = np.flip(arr.reshape(h, w, c), 0)[:, :, :3][:, :, ::-1] if c >= 3 else np.flip(arr.reshape(h, w, 1), 0)
            if img.ndim == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise RuntimeError(f"unexpected buffer size n={n} vs {w}x{h}")

    if not (img.flags["C_CONTIGUOUS"] and img.flags["WRITEABLE"]):
        img = np.ascontiguousarray(img.copy())
    return img


def _read_sensor_once(sim, sensor: int, stepped: int) -> np.ndarray:
    """Read one frame from a vision sensor (handles explicit handling and stepped sim)."""
    try:
        if bool(sim.getObjectInt32Param(sensor, sim.visionintparam_explicit_handling)):
            sim.handleVisionSensor(sensor)
    except Exception:
        pass

    if stepped and hasattr(sim, "setStepping"):
        try:
            sim.step()
        except Exception:
            pass

    # Newer API
    try:
        img, res = sim.getVisionSensorImg(sensor)  # bytes, [w,h]
        w, h = int(res[0]), int(res[1])
        return _decode_img(img, w, h)
    except Exception as e1:
        # Fallback signatures
        try:
            out = sim.getVisionSensorImage(sensor)
            if isinstance(out, (list, tuple)) and len(out) == 3 and isinstance(out[1], (int, float)):
                img, w, h = out
                w, h = int(w), int(h)
            else:
                img, res = out
                w, h = int(res[0]), int(res[1])

            if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
                arr = np.asarray(img, dtype=np.float32)
                img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
            return _decode_img(img, w, h)
        except Exception as e2:
            raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")


def encode_jpeg(bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


# ====================== FrameGrabber (snapshot source) ======================

class FrameGrabber:
    """
    Background thread that keeps the latest frame from a CoppeliaSim vision sensor.
    Use .start() once, then .get_frame() to fetch a copy.
    """
    def __init__(self, host: str, port: int, sensor_alias: str, stepped: int = 1, fps: int = 15):
        self.host = host
        self.port = port
        self.sensor_alias = sensor_alias
        self.stepped = int(stepped)
        self.period = 1.0 / max(1, int(fps))

        self.client = None
        self.sim = None
        self.sensor = None

        self._last = None  # (frame, ts)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _connect(self):
        self.client = RemoteAPIClient(self.host, self.port)
        self.sim = self.client.require("sim")
        self.sensor = _resolve_sensor(self.sim, self.sensor_alias)
        if self.sensor is None:
            raise RuntimeError(f"Vision sensor '{self.sensor_alias}' not found")

        # Ensure simulation is running
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        # Enable stepping if requested
        if self.stepped and hasattr(self.sim, "setStepping"):
            try:
                self.sim.setStepping(True)
            except Exception:
                pass

    def _loop(self):
        sleep_dt = self.period * 0.5
        while not self._stop.is_set():
            try:
                if self.sim is None:
                    self._connect()
                frame = _read_sensor_once(self.sim, self.sensor, self.stepped)
                with self._lock:
                    self._last = (frame, time.time())
            except Exception as e:
                print(f"[csim_frames][ERR] snapshot capture: {e}", flush=True)
                time.sleep(0.2)
            time.sleep(max(0.0, sleep_dt))

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last is None:
                return None
            return self._last[0].copy()


# ====================== MJPEG generator for raw streams ======================

def mjpeg_generator_raw(host: str, port: int, alias: str, fps: int = 15, stepped: int = 0):
    """
    Flask streaming generator for /stream_raw/<which>.
    Connects to CoppeliaSim, reads given sensor alias each loop, and yields MJPEG frames.
    """
    period = 1.0 / max(1, int(fps))
    try:
        client = RemoteAPIClient(host, port)
        sim = client.require("sim")
        sensor = _resolve_sensor(sim, alias)
        if sensor is None:
            raise RuntimeError(f"Vision sensor '{alias}' not found")

        # Make sure sim is running
        try:
            st = sim.getSimulationState()
            if st in (sim.simulation_stopped, sim.simulation_paused):
                sim.startSimulation()
        except Exception:
            pass

        if stepped and hasattr(sim, "setStepping"):
            try:
                sim.setStepping(True)
            except Exception:
                pass

    except Exception as e:
        # Emit a single error frame
        msg = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(msg, f"RAW connect error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_jpeg(msg) + b"\r\n")
        return

    while True:
        try:
            frame = _read_sensor_once(sim, sensor, stepped)
            jpg = encode_jpeg(frame)
        except Exception as e:
            err = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(err, f"RAW error: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            jpg = encode_jpeg(err)
            time.sleep(0.2)

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(period)
