# app.py
import time
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
from flask import Flask, Response, render_template_string

# ----------------------------
# Config you might want to tweak
# ----------------------------
ZMQ_HOST = "127.0.0.1"
ZMQ_PORT = 23000
LEGACY_HOST = "127.0.0.1"
LEGACY_PORT = 19997
VISION_SENSOR_CANDIDATES = ("/Vision_sensor", "Vision_sensor", "/VisionSensor", "VisionSensor")

HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>CoppeliaSim Vision Sensor Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:0;padding:2rem;background:#0b0c10;color:#e5e7eb}
      .wrap{max-width:960px;margin:0 auto}
      h1{font-weight:700;margin-bottom:1rem}
      .panel{background:#111827;border:1px solid #1f2937;border-radius:16px;padding:1rem}
      img{width:100%;height:auto;border-radius:12px;display:block}
      .meta{font-size:.9rem;color:#9ca3af;margin:.75rem 0 0}
      code{background:#1f2937;padding:.15rem .35rem;border-radius:6px}
      a{color:#93c5fd}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>CoppeliaSim Vision Sensor Stream</h1>
      <div class="panel">
        <img src="/stream.mjpg" alt="Vision Sensor Stream" />
        <div class="meta">
          If the image is black or frozen, make sure the simulation is running in CoppeliaSim and the vision sensor exists.
        </div>
      </div>
    </div>
  </body>
</html>"""

# ----------------------------
# Backends
# ----------------------------

class BaseBackend:
    def connect(self): ...
    def get_frame(self) -> Optional[np.ndarray]: ...
    def close(self): ...

# ---- ZeroMQ Remote API (preferred) ----
import os
import numpy as np
import cv2

class ZMQBackend(BaseBackend):
    def __init__(self, host=ZMQ_HOST, port=ZMQ_PORT, candidates=VISION_SENSOR_CANDIDATES):
        self.host = host
        self.port = port
        self.candidates = candidates
        self.client = None
        self.sim = None
        self.handle = None
        # optional: override with exact path
        self.preferred_path = os.environ.get("VISION_SENSOR_PATH")  # e.g. /Robot/vision_sensor

    def connect(self):
        try:
            try:
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            except ImportError:
                from zmqRemoteApi import RemoteAPIClient  # type: ignore

            self.client = RemoteAPIClient(self.host, self.port)
            self.sim = self.client.getObject("sim")

            self.handle = self.resolve_vision_sensor()
            if self.handle is None:
                raise RuntimeError("No Vision sensor found in the scene.")

        except Exception as e:
            raise RuntimeError(f"ZMQ connect failed: {e}")

    def resolve_vision_sensor(self):
        # 1) exact override
        if self.preferred_path:
            try:
                h = self.sim.getObject(self.preferred_path)
                print(f"[ZMQ] Using preferred vision sensor: {self.preferred_path}")
                return h
            except Exception as e:
                print(f"[ZMQ] Preferred VISION_SENSOR_PATH not found: {e}")

        # 2) try common names provided
        for name in self.candidates:
            try:
                h = self.sim.getObject(name)
                print(f"[ZMQ] Found vision sensor by name: {name}")
                return h
            except Exception:
                pass

        # 3) enumerate by type (most reliable)
        sensors = []
        try:
            # Fast path in recent CoppeliaSim versions
            sensors = self.sim.getObjects(self.sim.object_visionsensor_type)
        except Exception:
            # Fallback: iterate all and filter by type
            try:
                all_objs = self.sim.getObjects(self.sim.object_all_type)
                sensors = [h for h in all_objs if self.sim.getObjectType(h) == self.sim.object_visionsensor_type]
            except Exception as e:
                print(f"[ZMQ] Could not enumerate objects: {e}")
                sensors = []

        if not sensors:
            print("[ZMQ] No vision sensors found via enumeration.")
            return None

        print("[ZMQ] Vision sensors detected:")
        for h in sensors:
            try:
                # 1 means 'alias including full path'
                alias = self.sim.getObjectAlias(h, 1)
            except Exception:
                alias = f"<handle {h}>"
            print(f"   - {alias}")
        print("      (Set env VISION_SENSOR_PATH to one of the above to pick a specific sensor.)")

        return sensors[0]  # choose the first by default

    def get_frame(self):
        try:
            # Newer API: getVisionSensorImg; fallback: getVisionSensorCharImage
            if hasattr(self.sim, "getVisionSensorImg"):
                out = self.sim.getVisionSensorImg(self.handle)
                if isinstance(out, (list, tuple)) and len(out) == 3:
                    img_bytes, resX, resY = out
                elif isinstance(out, (list, tuple)) and len(out) == 2:
                    img_bytes, res = out
                    resX, resY = int(res[0]), int(res[1])
                else:
                    raise RuntimeError("Unexpected getVisionSensorImg return format")
            elif hasattr(self.sim, "getVisionSensorCharImage"):
                img_bytes, res = self.sim.getVisionSensorCharImage(self.handle)
                resX, resY = int(res[0]), int(res[1])
            else:
                raise RuntimeError("No vision image function found in ZMQ API")

            arr = np.frombuffer(bytearray(img_bytes), dtype=np.uint8)
            if arr.size != resX * resY * 3:
                return None
            frame = arr.reshape((resY, resX, 3))
            frame = np.flipud(frame)                # bottom-left origin -> top-left
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            return None

    def close(self):
        pass
    def __init__(self, host=ZMQ_HOST, port=ZMQ_PORT, candidates=VISION_SENSOR_CANDIDATES):
        self.host = host
        self.port = port
        self.candidates = candidates
        self.client = None
        self.sim = None
        self.handle = None

    def connect(self):
        try:
            try:
                # pip package
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            except ImportError:
                # module bundled with CoppeliaSim
                from zmqRemoteApi import RemoteAPIClient  # type: ignore
            self.client = RemoteAPIClient(self.host, self.port)
            self.sim = self.client.getObject("sim")
            # resolve handle
            last_err = None
            for name in self.candidates:
                try:
                    self.handle = self.sim.getObject(name)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if self.handle is None:
                raise RuntimeError(f"Vision sensor not found. Tried: {self.candidates}. Last error: {last_err}")
        except Exception as e:
            raise RuntimeError(f"ZMQ connect failed: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Returns a numpy uint8 BGR image or None on failure.
        """
        try:
            # Many CoppeliaSim builds provide either getVisionSensorImg or getVisionSensorCharImage.
            if hasattr(self.sim, "getVisionSensorImg"):
                # Expected returns: (bytes, resX, resY) or (bytes, [resX,resY])
                out = self.sim.getVisionSensorImg(self.handle)
                # normalize shapes across versions:
                if isinstance(out, (list, tuple)) and len(out) == 3:
                    img_bytes, resX, resY = out
                elif isinstance(out, (list, tuple)) and len(out) == 2:
                    img_bytes, res = out
                    resX, resY = int(res[0]), int(res[1])
                else:
                    # Unexpected; let it raise to trigger fallback/None
                    raise RuntimeError("Unexpected getVisionSensorImg return format")

            elif hasattr(self.sim, "getVisionSensorCharImage"):
                img_bytes, res = self.sim.getVisionSensorCharImage(self.handle)
                resX, resY = int(res[0]), int(res[1])
            else:
                raise RuntimeError("No vision image function found in ZMQ API")

            arr = np.frombuffer(bytearray(img_bytes), dtype=np.uint8)
            if arr.size != resX * resY * 3:
                return None
            frame = arr.reshape((resY, resX, 3))

            # CoppeliaSim returns images with origin at bottom-left; flip vertically:
            frame = np.flipud(frame)

            # Convert RGB -> BGR for OpenCV/JPEG
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            return None

    def close(self):
        # ZMQ client doesn't need explicit close typically
        pass

# ---- Legacy Remote API (simx_*) ----
class LegacyBackend(BaseBackend):
    def __init__(self, host=LEGACY_HOST, port=LEGACY_PORT, candidates=VISION_SENSOR_CANDIDATES):
        self.host = host
        self.port = port
        self.candidates = candidates
        self.client_id = -1
        self.sim = None
        self.handle = None
        self._stream_started = False

    def connect(self):
        try:
            import sim  # legacy remote API helper (shippped with CoppeliaSim)
            self.sim = sim

            # close any existing just in case
            self.sim.simxFinish(-1)
            self.client_id = self.sim.simxStart(self.host, self.port, True, True, 5000, 5)
            if self.client_id == -1:
                raise RuntimeError("Cannot connect to legacy Remote API server")

            # resolve object handle (try multiple names/paths)
            last_err = None
            for name in self.candidates:
                rc, h = self.sim.simxGetObjectHandle(self.client_id, name, self.sim.simx_opmode_blocking)
                if rc == self.sim.simx_return_ok:
                    self.handle = h
                    break
                last_err = f"rc={rc}"
            if self.handle is None:
                raise RuntimeError(f"Vision sensor not found via legacy API. Tried: {self.candidates}. Last: {last_err}")

            # Prime the stream:
            rc, res, img = self.sim.simxGetVisionSensorImage(self.client_id, self.handle, 0, self.sim.simx_opmode_streaming)
            self._stream_started = True
        except Exception as e:
            raise RuntimeError(f"Legacy connect failed: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._stream_started:
            return None
        try:
            rc, res, img = self.sim.simxGetVisionSensorImage(
                self.client_id, self.handle, 0, self.sim.simx_opmode_buffer
            )
            if rc not in (self.sim.simx_return_ok,):
                # not ready yet
                return None
            w, h = int(res[0]), int(res[1])
            arr = np.array(img, dtype=np.uint8)
            if arr.size != w * h * 3:
                return None
            frame = arr.reshape((h, w, 3))
            # Legacy returns bottom-to-top too:
            frame = cv2.flip(frame, 0)
            # Legacy returns RGB; convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            return None

    def close(self):
        try:
            if self.client_id != -1 and self.sim is not None:
                self.sim.simxFinish(self.client_id)
        except Exception:
            pass


# ----------------------------
# Streamer (backend-agnostic)
# ----------------------------
class VisionStreamer:
    def __init__(self):
        self.backend: Optional[BaseBackend] = None
        self.latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        # Try ZMQ first:
        for Backend in (ZMQBackend, LegacyBackend):
            try:
                b = Backend()
                b.connect()
                self.backend = b
                print(f"[stream] Using backend: {Backend.__name__}")
                break
            except Exception as e:
                print(f"[stream] {Backend.__name__} not available: {e}")
        if self.backend is None:
            raise RuntimeError("Could not connect to CoppeliaSim via ZMQ or Legacy API.")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # Pull frames continuously and encode to JPEG
        smoothing_fps = 25.0
        frame_interval = 1.0 / smoothing_fps
        next_t = time.time()

        while not self._stop.is_set():
            frame = self.backend.get_frame()
            if frame is not None:
                # Optional: you could overlay FPS/time here
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with self._lock:
                        self.latest_jpeg = jpeg.tobytes()
            # pace a bit to avoid pegging CPU
            now = time.time()
            sleep_for = max(0.0, next_t - now)
            time.sleep(sleep_for)
            next_t = max(now, next_t) + frame_interval

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.backend:
            self.backend.close()


streamer = VisionStreamer()
streamer.start()

# ----------------------------
# Flask app (MJPEG stream)
# ----------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stream.mjpg")
def stream():
    def gen():
        boundary = b"--frame"
        while True:
            jpeg = streamer.get_jpeg()
            if jpeg is None:
                # If no frame yet, send a tiny black JPEG to keep the client alive
                blank = np.zeros((2, 2, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", blank)
                payload = buf.tobytes() if ok else b""
            else:
                payload = jpeg
            yield (
                boundary + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
                + payload + b"\r\n"
            )
            # Clients typically pull at their own rate; short sleep is fine
            time.sleep(0.001)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    try:
        # threaded=True allows multiple viewers
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        streamer.stop()
