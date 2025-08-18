#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-setup vision web server for CoppeliaSim:
- Connects to /visionSensor and /mat by default
- Streams Raw + 5 model overlays in a browser dashboard
- Loads model wrappers from ML_detection/<stem>.py next to <stem>.pth
- Projects mat-frame polylines into the camera view

Run with no args:
  ros2 run suture_arm vision_web
or:
  python3 vision_web.py

Env overrides (no CLI flags needed):
  CSIM_HOST=127.0.0.1
  CSIM_PORT=23001
  VISION_SENSOR=/visionSensor
  MAT_ALIAS=/mat
  PORT=8000
  FPS=100
  QUALITY=95
  SCALE=1.25
  MODELS="mask_cuts_detector.pth,cuts_detector_best.pth,mask_cuts_detector2.pth,cuts_detector_best2.pth,mask_cuts_detector3.pth"
"""

import os
import time
import math
import importlib.util
import traceback
import threading
import numpy as np
import cv2

from flask import Flask, Response, send_from_directory, render_template_string, make_response, jsonify

# ZMQ Remote API client
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback

# ROS2 (background cuts listener; optional)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


# ---------------- Defaults & Env ----------------
CSIM_HOST      = os.getenv("CSIM_HOST", "127.0.0.1")
CSIM_PORT      = int(os.getenv("CSIM_PORT", "23001"))  # your server is on 23001
SENSOR_ALIAS   = os.getenv("VISION_SENSOR", "/visionSensor")
MAT_ALIAS      = os.getenv("MAT_ALIAS", "/mat")
PORT           = int(os.getenv("PORT", "8000"))
FPS            = int(os.getenv("FPS", "10"))
JPEG_QUALITY   = int(os.getenv("QUALITY", "95"))
SERVER_SCALE   = float(os.getenv("SCALE", "1.25"))   # 1.0 = off
MODELS_CSV     = os.getenv("MODELS",
    "mask_cuts_detector.pth,cuts_detector_best.pth,mask_cuts_detector2.pth,cuts_detector_best2.pth,mask_cuts_detector3.pth"
)
MODEL_PTHS     = [s.strip() for s in MODELS_CSV.split(",") if s.strip()]


# ---------------- HTML ----------------
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>UR3 Vision — Suture Models Dashboard</title>
<style>
 :root{ --gap:18px; --cardpad:12px }
 *{box-sizing:border-box}
 body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}
 header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 nav a{margin-right:12px}
 .meta{color:#666;font-size:0.95rem;margin:6px 0 16px}
 .grid{
   display:grid;
   grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
   gap: var(--gap);
   align-items:start;
 }
 .card{border:1px solid #eaeaea;border-radius:14px;box-shadow:0 2px 10px rgba(0,0,0,.06);padding:var(--cardpad)}
 .title{font-weight:600;margin:0 0 8px}
 .stream{width:100%;height:auto;border:1px solid #ddd;border-radius:10px}
 .small{font-size:.85rem;color:#777}
 .btns a{display:inline-block;margin-right:8px;margin-top:8px;font-size:.9rem;text-decoration:none}
 code{background:#f4f4f4;padding:2px 6px;border-radius:6px}
</style></head><body>
<header>
  <h2>UR3 Vision — Live & Model Overlays</h2>
  <nav>
    <a href="/">Dashboard</a>
    <a href="/models">Model files</a>
    <a href="/info">Info</a>
  </nav>
</header>

<div class="meta">
  Sensor <code>{{sensor}}</code> • FPS {{fps}} • Mat <code>{{mat}}</code> • JPEG Q {{quality}} • Scale {{scale}}x
</div>

<div class="grid">
  <div class="card">
    <div class="title">Raw feed</div>
    <img class="stream" src="/video_raw" alt="raw stream"/>
    <div class="btns"><a href="/snapshot_raw">📸 snapshot</a></div>
  </div>

  {% for key, label in model_cards %}
  <div class="card">
    <div class="title">{{label}}</div>
    <img class="stream" src="/video_model/{{key}}" alt="{{label}} stream"/>
    <div class="small">source: {{key}}</div>
    <div class="btns">
      <a href="/snapshot_model/{{key}}">📸 snapshot</a>
    </div>
  </div>
  {% endfor %}
</div>
</body></html>
"""

MODELS_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Detector model files</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}
 table{border-collapse:collapse;min-width:620px}
 td,th{border:1px solid #ddd;padding:8px} th{background:#f7f7f7;text-align:left}
 code{background:#f2f2f2;padding:2px 4px;border-radius:4px}
 a{margin-right:12px}
</style></head><body>
<h2>Detector model files</h2>
<p>Directory: <code>{{root}}</code></p>
{% if files %}
<table><tr><th>Model (.pth)</th><th>Wrapper (.py)</th><th>Size</th><th>Download</th></tr>
{% for row in files %}
<tr>
  <td>{{row.pth}}</td>
  <td>{{row.py or "-"}}</td>
  <td>{{row.size}}</td>
  <td><a href="/models/{{row.pth}}">download</a>{% if row.py %} · <a href="/models/{{row.py}}">wrapper</a>{% endif %}</td>
</tr>
{% endfor %}</table>
{% else %}<p><em>No *.pth files found.</em></p>{% endif %}
<p><a href="/">⟵ Back</a></p></body></html>
"""

def sizeof_fmt(num):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


# ---------- Cuts buffer (thread-safe, for optional ROS topic) ----------
class CutsBuffer:
    def __init__(self): self._lock = threading.RLock(); self._data = None
    def update(self, data_dict):  # not used by overlays; kept for debugging
        with self._lock: self._data = data_dict
    def get(self):
        with self._lock: return None if self._data is None else dict(self._data)


class CutsNode(Node):
    def __init__(self, buf: CutsBuffer):
        super().__init__('vision_web_cuts_listener')
        self.buf = buf
        self.sub = self.create_subscription(StringMsg, '/suture_cuts', self.on_msg, 10)
        self.get_logger().info('Listening to /suture_cuts')
    def on_msg(self, msg: StringMsg):
        try:
            import json
            d = json.loads(msg.data)
            self.buf.update(d)
        except Exception as e:
            self.get_logger().warn(f'Failed to parse /suture_cuts JSON: {e}')


def start_cuts_listener_in_thread(buf: CutsBuffer):
    def _run():
        rclpy.init(args=None)
        node = CutsNode(buf)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


# ---------- Polyline & overlay helpers ----------
def sample_polyline(poly: np.ndarray, spacing: float) -> np.ndarray:
    if poly.shape[0] < 2:
        return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    total = s[-1]
    if total < 1e-9:
        return poly[:1].copy()
    t = np.arange(0, total, max(spacing, 1e-6))
    x = np.interp(t, s, poly[:, 0])
    y = np.interp(t, s, poly[:, 1])
    res = np.c_[x, y]
    if (total - t[-1]) > 1e-6:
        res = np.vstack([res, poly[-1]])
    return res

def plan_bites_2d(poly2d: np.ndarray, spacing: float, bite: float):
    pts = sample_polyline(poly2d, spacing)
    entries, exits = [], []
    for i, p in enumerate(pts):
        if i == 0:
            t = pts[min(1, len(pts)-1)] - pts[0]
        elif i == len(pts)-1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i+1] - pts[i-1]
        n = np.array([-t[1], t[0]])
        nt = np.linalg.norm(n)
        if nt < 1e-9: continue
        n /= nt
        entries.append(p + 0.5 * bite * n)
        exits.append (p - 0.5 * bite * n)
    if not entries:
        return np.empty((0,2)), np.empty((0,2)), np.empty((0,2))
    return pts, np.vstack(entries), np.vstack(exits)


# ---------- Vision streamer ----------
class VisionStreamer:
    """Thread-safe CoppeliaSim vision sensor streamer with projection helpers."""
    def __init__(self, host, port, sensor=SENSOR_ALIAS, fps=FPS, mat=MAT_ALIAS, jpg_quality=JPEG_QUALITY, server_scale=SERVER_SCALE):
        self.lock = threading.RLock()
        self.client = RemoteAPIClient(host, port)
        self.sim = self.client.require('sim')
        self.fps = int(max(1, fps))
        self.sensor_name = sensor
        self.mat_alias = mat
        self.jpg_q = int(np.clip(jpg_quality, 60, 100))
        self.scale = float(max(0.1, server_scale))
        self.last_res = (0, 0)

        # stepped mode
        with self.lock:
            try:
                if hasattr(self.sim, 'setStepping'): self.sim.setStepping(True)
            except Exception: pass

        # sensor handle
        self.sensor = None
        for cand in (sensor, sensor.lstrip('/'), sensor + '#0'):
            try:
                with self.lock: self.sensor = self.sim.getObject(cand)
                self.sensor_name = cand; break
            except Exception: pass
        if self.sensor is None:
            raise RuntimeError(f"vision sensor '{sensor}' not found")

        # mat handle (optional)
        self.mat = None
        try:
            with self.lock: self.mat = self.sim.getObject(mat)
        except Exception:
            print(f"[vision_web] WARN: mat alias '{mat}' not found; overlay assumes origin")

        # explicit handling?
        self.explicit = False
        try:
            with self.lock:
                self.explicit = bool(self.sim.getObjectInt32Param(
                    self.sensor, self.sim.visionintparam_explicit_handling))
        except Exception: pass

        # ensure sim running
        try:
            with self.lock:
                st = self.sim.getSimulationState()
                if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                    self.sim.startSimulation()
        except Exception: pass

        # perspective params
        self.perspective = True
        self.vfov = None
        try:
            with self.lock:
                self.perspective = bool(self.sim.getObjectInt32Param(self.sensor, self.sim.visionintparam_perspective_mode))
            if self.perspective:
                with self.lock:
                    self.vfov = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_perspective_angle))
        except Exception: pass

        # warmup
        for _ in range(5):
            self._handle_if_needed(); self._step(); time.sleep(0.01)

    def _step(self):
        try:
            with self.lock: self.sim.step()
        except Exception: pass

    def _handle_if_needed(self):
        if self.explicit:
            try:
                with self.lock: self.sim.handleVisionSensor(self.sensor)
            except Exception: pass

    def _read_raw_image(self):
        try:
            with self.lock: img, res = self.sim.getVisionSensorImg(self.sensor)
            resX, resY = int(res[0]), int(res[1])
        except Exception:
            try:
                with self.lock: img, resX, resY = self.sim.getVisionSensorCharImage(self.sensor)
                resX, resY = int(resX), int(resY)
            except Exception as e1:
                try:
                    with self.lock: out = self.sim.getVisionSensorImage(self.sensor)
                    if isinstance(out, (list, tuple)) and len(out)==3:
                        img, resX, resY = out; resX, resY = int(resX), int(resY)
                    else:
                        img, res = out; resX, resY = int(res[0]), int(res[1])
                    img = (np.array(img, dtype=np.float32)*255).clip(0,255).astype(np.uint8)
                    return img, resX, resY
                except Exception as e2:
                    raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")
        if isinstance(img, (bytes, bytearray)):
            arr = np.frombuffer(img, dtype=np.uint8)
        else:
            arr = np.array(img)
            if arr.dtype!=np.uint8:
                arr = (arr.astype(np.float32).clip(0,255)).astype(np.uint8)
        return arr, resX, resY

    def _frame_from_arr(self, arr, w, h):
        n = arr.size
        if n == w*h:      # gray
            frame = arr.reshape(h,w); frame = np.flip(frame,0); frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif n == w*h*3:  # rgb
            frame = arr.reshape(h,w,3); frame = np.flip(frame,0)[:, :, ::-1]
        elif n == w*h*4:  # rgba
            frame = arr.reshape(h,w,4); frame = np.flip(frame,0)[:, :, :3][:, :, ::-1]
        else:
            if n%(w*h)==0:
                c = n//(w*h)
                frame = arr.reshape(h,w,c); frame = np.flip(frame,0)
                if c==1: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif c>=3: frame = frame[:, :, :3][:, :, ::-1]
            else:
                raise RuntimeError(f"unexpected buffer size: n={n}, w={w}, h={h}")
        return frame

    def grab_raw_frame(self):
        for _ in range(10):
            self._handle_if_needed()
            arr, w, h = self._read_raw_image()
            if w>0 and h>0 and arr.size>=w*h:
                frame = self._frame_from_arr(arr, w, h)
                self.last_res = (w, h)
                if self.scale!=1.0:
                    new_w = max(1, int(round(frame.shape[1]*self.scale)))
                    new_h = max(1, int(round(frame.shape[0]*self.scale)))
                    frame = cv2.resize(frame, (new_w,new_h), interpolation=cv2.INTER_CUBIC)
                return frame
            self._step(); time.sleep(0.01)
        raise RuntimeError(f"vision sensor not ready (w={w}, h={h}, n={arr.size})")

    @staticmethod
    def _M12_to_T(m12):
        T = np.eye(4, dtype=float); T[:3,:4] = np.array(m12, dtype=float).reshape(3,4); return T

    def get_T_cam_mat(self):
        if self.mat is None: return np.eye(4, dtype=float)
        with self.lock:
            M = self.sim.getObjectMatrix(self.sensor, self.mat)  # mat->cam
        return self._M12_to_T(M)

    def get_intrinsics(self, w, h):
        if self.perspective and (self.vfov is None or not np.isfinite(self.vfov)):
            try:
                with self.lock:
                    self.vfov = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_perspective_angle))
            except Exception:
                self.vfov = math.radians(60.0)
        cx, cy = 0.5*w, 0.5*h
        if self.perspective:
            fy = (h*0.5)/math.tan(0.5*(self.vfov or math.radians(60.0)))
            fx = fy*(w/h)
            return fx, fy, cx, cy, True
        else:
            try:
                with self.lock:
                    ortho = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_ortho_size))
            except Exception:
                ortho = 0.2
            mpp = (2.0*ortho)/h
            return mpp, mpp, cx, cy, False

    def project_pts(self, P_cam, w, h):
        fx, fy, cx, cy, persp = self.get_intrinsics(w, h)
        uv = []
        for x,y,z in P_cam:
            if persp:
                if z<=1e-6: uv.append([np.nan,np.nan]); continue
                u = fx*(x/z)+cx; v = -fy*(y/z)+cy
            else:
                u = (x/(2.0*fx))*w + cx; v = (-y/(2.0*fy))*h + cy
            uv.append([u,v])
        return np.array(uv)

    def draw_overlay(self, frame_bgr, cuts_dict, color_cut=(60,220,60), color_bite=(255,180,60)):
        if cuts_dict is None: return frame_bgr
        H, W = frame_bgr.shape[:2]
        spacing = float(cuts_dict.get('params', {}).get('spacing', 0.006))
        bite    = float(cuts_dict.get('params', {}).get('bite',    0.005))
        thick = max(1, int(round(min(W,H)/400.0)))

        T = self.get_T_cam_mat()

        def to3(p2): return np.c_[p2, np.zeros((len(p2),1))]
        def xform(P):
            if P.size==0: return P
            P_h = np.c_[P, np.ones((len(P),1))]
            return (T @ P_h.T).T[:, :3]

        for cut in cuts_dict.get('cuts', []):
            poly = np.array(cut.get('polyline', []), dtype=float)
            if poly.size==0: continue
            poly_xy = poly[:, :2] if poly.shape[1] >= 2 else poly

            pts, entries, exits = plan_bites_2d(poly_xy, spacing, bite)
            P_cut   = to3(poly_xy)
            P_entry = to3(entries) if len(entries) else np.empty((0,3))
            P_exit  = to3(exits)   if len(exits)   else np.empty((0,3))

            C_cut   = xform(P_cut)
            C_entry = xform(P_entry)
            C_exit  = xform(P_exit)

            U_cut   = self.project_pts(C_cut,  W, H)
            U_entry = self.project_pts(C_entry,W, H)
            U_exit  = self.project_pts(C_exit, W, H)

            pts_int = [(int(round(u)), int(round(v))) for u,v in U_cut if np.isfinite(u) and np.isfinite(v)]
            if len(pts_int)>=2:
                cv2.polylines(frame_bgr, [np.array(pts_int, np.int32)], False, color_cut, thick, cv2.LINE_AA)
            for (ue,ve),(ux,vx) in zip(U_entry, U_exit):
                if not (np.isfinite(ue) and np.isfinite(ve) and np.isfinite(ux) and np.isfinite(vx)): continue
                p1=(int(round(ue)),int(round(ve))); p2=(int(round(ux)),int(round(vx)))
                cv2.line(frame_bgr, p1, p2, color_bite, thick, cv2.LINE_AA)
                cv2.circle(frame_bgr, p1, max(2,thick+1), (0,200,255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, p2, max(2,thick+1), (0,128,255), -1, cv2.LINE_AA)
        return frame_bgr


# ---------- Detector plugin manager ----------
class DetectorPlugin:
    """Loads ML_detection/<stem>.py next to <stem>.pth and runs predict(handle, frame_bgr)."""
    def __init__(self, models_root, pth_name, label=None, infer_every=2):
        self.models_root = models_root
        self.pth = pth_name
        self.label = label or os.path.splitext(pth_name)[0]
        self.stem = os.path.splitext(pth_name)[0]
        self.py  = self.stem + '.py'
        self.module = None
        self.handle = None
        self.last_cuts = None
        self.frame_count = 0
        self.infer_every = max(1, int(infer_every))
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        py_path = os.path.join(self.models_root, self.py)
        pth_path = os.path.join(self.models_root, self.pth)
        if not os.path.isfile(pth_path):
            print(f"[vision_web] Model file missing: {pth_path}")
            return
        if not os.path.isfile(py_path):
            print(f"[vision_web] Wrapper missing: {py_path}  (will show stream with note)")
            return
        spec = importlib.util.spec_from_file_location(self.stem, py_path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore
            self.module = mod
            self.handle = getattr(mod, 'load')(pth_path)
            print(f"[vision_web] Loaded model: {self.pth} via {self.py}")
        except Exception as e:
            print(f"[vision_web] ERROR loading {self.py}: {e}")
            self.module = None; self.handle=None

    def infer_maybe(self, frame_bgr):
        with self.lock:
            self.frame_count += 1
            if self.module is None or self.handle is None:
                return None
            if (self.frame_count % self.infer_every) != 0 and self.last_cuts is not None:
                return self.last_cuts
            try:
                pred = self.module.predict(self.handle, frame_bgr)  # must return cuts_dict
                self.last_cuts = pred
                return pred
            except Exception as e:
                print(f"[vision_web] ERROR in predict() for {self.stem}: {e}")
                return self.last_cuts


def find_models_root():
    # 1) installed share/suture_arm/ml
    if get_package_share_directory:
        try:
            share = get_package_share_directory('suture_arm')
            p = os.path.join(share, 'ml')
            if os.path.isdir(p): return p
        except Exception: pass
    # 2) dev tree fallbacks
    here = os.path.dirname(__file__)
    for cand in ('ML_detection', os.path.join('..', 'ML_detection')):
        p = os.path.abspath(os.path.join(here, cand))
        if os.path.isdir(p): return p
    # 3) cwd
    return os.getcwd()


# -------------- Flask app factory --------------
def create_app():
    vs = VisionStreamer(CSIM_HOST, CSIM_PORT, SENSOR_ALIAS, FPS, MAT_ALIAS, JPEG_QUALITY, SERVER_SCALE)
    app = Flask(__name__)

    # Prepare detector plugins
    root = find_models_root()
    plugins = [DetectorPlugin(root, pth, label=os.path.splitext(pth)[0], infer_every=2) for pth in MODEL_PTHS]
    plugin_map = {pl.stem: pl for pl in plugins}

    # Optional: tiny /suture_cuts watcher (handy for debugging)
    cuts_buf = CutsBuffer()
    start_cuts_listener_in_thread(cuts_buf)

    # ---------- Routes ----------
    @app.route('/')
    def index():
        model_cards = [(pl.stem, pl.label) for pl in plugins]
        return render_template_string(INDEX_HTML,
                                      sensor=vs.sensor_name, fps=vs.fps, mat=vs.mat_alias,
                                      quality=vs.jpg_q, scale=vs.scale,
                                      model_cards=model_cards)

    @app.route('/video_raw')
    def video_raw():
        def gen():
            delay = 1.0 / vs.fps
            while True:
                try:
                    vs._step()
                    frame = vs.grab_raw_frame()
                    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), vs.jpg_q])
                    if not ok: time.sleep(delay); continue
                    chunk = jpg.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                           b'Cache-Control: no-cache\r\nPragma: no-cache\r\n'
                           b'Content-Length: ' + str(len(chunk)).encode() + b'\r\n\r\n' +
                           chunk + b'\r\n')
                except Exception as e:
                    print('[vision_web] MJPEG raw error:', e); time.sleep(0.25)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/video_model/<stem>')
    def video_model(stem):
        pl = plugin_map.get(stem)
        if pl is None:
            return make_response(f"Unknown model: {stem}", 404)
        def gen():
            delay = 1.0 / vs.fps
            while True:
                try:
                    vs._step()
                    frame = vs.grab_raw_frame()
                    cuts = pl.infer_maybe(frame)
                    if cuts is None and (pl.module is None or pl.handle is None):
                        cv2.putText(frame, f"Wrapper not found for {pl.py}",
                                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5,5,240), 2, cv2.LINE_AA)
                    else:
                        frame = vs.draw_overlay(frame, cuts)
                    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), vs.jpg_q])
                    if not ok: time.sleep(delay); continue
                    chunk = jpg.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                           b'Cache-Control: no-cache\r\nPragma: no-cache\r\n'
                           b'Content-Length: ' + str(len(chunk)).encode() + b'\r\n\r\n' +
                           chunk + b'\r\n')
                except Exception as e:
                    print(f'[vision_web] MJPEG model {stem} error:', e); time.sleep(0.25)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/snapshot_raw')
    def snapshot_raw():
        try:
            frame = vs.grab_raw_frame()
            ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(vs.jpg_q, 92)])
            if not ok: raise RuntimeError('encode failed')
            resp = make_response(jpg.tobytes())
            resp.headers['Content-Type'] = 'image/jpeg'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        except Exception as e:
            tb = traceback.format_exc()
            return make_response(f"SNAPSHOT RAW ERROR:\n{e}\n\n{tb}", 500)

    @app.route('/snapshot_model/<stem>')
    def snapshot_model(stem):
        pl = plugin_map.get(stem)
        if pl is None:
            return make_response(f"Unknown model: {stem}", 404)
        try:
            frame = vs.grab_raw_frame()
            cuts = pl.infer_maybe(frame)
            if cuts is None and (pl.module is None or pl.handle is None):
                cv2.putText(frame, f"Wrapper not found for {pl.py}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5,5,240), 2, cv2.LINE_AA)
            else:
                frame = vs.draw_overlay(frame, cuts)
            ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(vs.jpg_q, 95)])
            if not ok: raise RuntimeError('encode failed')
            resp = make_response(jpg.tobytes())
            resp.headers['Content-Type'] = 'image/jpeg'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        except Exception as e:
            tb = traceback.format_exc()
            return make_response(f"SNAPSHOT MODEL ERROR ({stem}):\n{e}\n\n{tb}", 500)

    # Files / info
    @app.route('/models')
    def models_page():
        root = find_models_root()
        rows = []
        for f in sorted(os.listdir(root)):
            if f.lower().endswith('.pth'):
                stem = os.path.splitext(f)[0]
                py = stem + '.py'
                rows.append({
                    "pth": f,
                    "py": py if os.path.isfile(os.path.join(root, py)) else None,
                    "size": sizeof_fmt(os.path.getsize(os.path.join(root, f)) if os.path.exists(os.path.join(root, f)) else 0)
                })
        return render_template_string(MODELS_HTML, root=root, files=rows)

    @app.route('/models/<path:fname>')
    def download_model(fname):
        root = find_models_root()
        return send_from_directory(root, fname, as_attachment=True)

    @app.route('/info')
    def info():
        w, h = vs.last_res
        return jsonify({
            "sensor": vs.sensor_name,
            "mat": vs.mat_alias,
            "last_frame_resolution": {"width": w, "height": h},
            "jpeg_quality": vs.jpg_q,
            "server_scale": vs.scale,
            "perspective": bool(vs.perspective),
            "vfov_deg": None if vs.vfov is None else float(np.degrees(vs.vfov)),
            "models_root": find_models_root(),
            "models": MODEL_PTHS,
        })

    return app


# -------------- entry point --------------
def main():
    # start background /suture_cuts listener (safe even if never used)
    start_cuts_listener_in_thread(CutsBuffer())

    app = create_app()
    print(f"[vision_web] Dashboard on http://0.0.0.0:{PORT}/")
    print(f"  CoppeliaSim {CSIM_HOST}:{CSIM_PORT}")
    print(f"  sensor={SENSOR_ALIAS}, mat={MAT_ALIAS}, models={MODEL_PTHS}")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()
