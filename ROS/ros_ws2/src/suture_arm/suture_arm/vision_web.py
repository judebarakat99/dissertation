#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
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

# ROS2 subscriber to pick up /suture_cuts
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

# ROS ament share lookup (optional)
try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


# ---------------- HTML ----------------
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>UR3 Vision — Suture Pad</title>
<style>
 :root { --card-pad:12px; --gap:16px; }
 body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}
 header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 nav a{margin-right:12px}
 .meta{color:#666;font-size:0.9rem;margin-bottom:12px}
 .row{display:flex;gap:var(--gap);flex-wrap:wrap;align-items:flex-start}
 .card{padding:var(--card-pad);border:1px solid #eee;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
 .title{font-weight:600;margin:0 0 8px}
 /* Make each stream about 40% of the viewport width */
 .stream{width:40vw;max-width:40vw;height:auto;border:1px solid #ddd;border-radius:8px}
 .btns a{display:inline-block;margin-right:8px;margin-top:8px;font-size:.9rem}
 @media (max-width:1100px){ .stream{width:46vw;max-width:46vw} }
 @media (max-width:780px){ .stream{width:95vw;max-width:95vw} }
</style></head><body>
<header>
  <h2>UR3 Vision — Top view</h2>
  <nav>
    <a href="/">Both</a>
    <a href="/snapshot">Overlay snapshot</a>
    <a href="/snapshot_raw">Raw snapshot</a>
    <a href="/models">Model files</a>
    <a href="/cuts">Cuts JSON</a>
    <a href="/info">Info</a>
  </nav>
</header>
<div class="meta">
  Sensor: <code>{{sensor}}</code> • FPS: {{fps}} • Mat: <code>{{mat}}</code> • JPEG Q: {{quality}} • Scale: {{scale}}x
</div>

<div class="row">
  <div class="card">
    <div class="title">Overlay (cuts + bites)</div>
    <img class="stream" src="/video" alt="overlay stream" />
    <div class="btns"><a href="/snapshot">📸 snapshot</a></div>
  </div>

  <div class="card">
    <div class="title">Raw feed</div>
    <img class="stream" src="/video_raw" alt="raw stream" />
    <div class="btns"><a href="/snapshot_raw">📸 snapshot</a></div>
  </div>
</div>
</body></html>
"""

MODELS_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Detector model files</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px}
 table{border-collapse:collapse;min-width:420px}
 td,th{border:1px solid #ddd;padding:8px} th{background:#f7f7f7;text-align:left}
 code{background:#f2f2f2;padding:2px 4px;border-radius:4px}
 a{margin-right:12px}
</style></head><body>
<h2>Detector model files</h2>
<p>Directory: <code>{{root}}</code></p>
{% if files %}
<table><tr><th>File</th><th>Size</th><th>Download</th></tr>
{% for f,s in files %}
<tr><td>{{f}}</td><td>{{s}}</td><td><a href="/models/{{f}}">download</a></td></tr>
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


# ---------- Cuts buffer (thread-safe) ----------
class CutsBuffer:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = None  # parsed dict from /suture_cuts

    def update(self, data_dict):
        with self._lock:
            self._data = data_dict

    def get(self):
        with self._lock:
            return None if self._data is None else dict(self._data)  # shallow copy


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


# ---------- Suture planning helpers (for overlay only) ----------
def sample_polyline(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Resample Nx2 polyline at ~equal spacing (m)."""
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


def plan_bites_2d(poly2d: np.ndarray, spacing: float, bite: float) -> tuple:
    """Return (cut_points, entry_points, exit_points) in mat XY (z=0)."""
    pts = sample_polyline(poly2d, spacing)  # Nx2
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
        if nt < 1e-9:
            continue
        n /= nt
        entries.append(p + 0.5 * bite * n)
        exits.append (p - 0.5 * bite * n)
    if len(entries) == 0:
        entries = np.empty((0, 2)); exits = np.empty((0, 2)); pts = np.empty((0, 2))
    else:
        entries = np.vstack(entries)
        exits  = np.vstack(exits)
    return pts, entries, exits


# ---------- Vision streamer with projection ----------
class VisionStreamer:
    """Thread-safe CoppeliaSim vision sensor streamer with overlay & warm-up."""
    def __init__(self, sensor_name='/visionSensor', fps=10, mat_alias='/mat',
                 cuts_buf: CutsBuffer = None, jpg_quality: int = 92, server_scale: float = 1.0):
        self.lock = threading.RLock()   # serialize ALL sim.* calls
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.fps = int(max(1, fps))
        self.sensor_name = sensor_name
        self.mat_alias = mat_alias
        self.cuts_buf = cuts_buf
        self.jpg_q = int(np.clip(jpg_quality, 60, 100))
        self.scale = float(max(0.1, server_scale))
        self.last_res = (0, 0)

        # Enable stepped mode
        with self.lock:
            try:
                if hasattr(self.sim, 'setStepping'):
                    self.sim.setStepping(True)
            except Exception as e:
                print('[vision_web] WARN setStepping failed:', e)

        # Resolve sensor
        self.sensor = None
        for cand in (sensor_name, sensor_name.lstrip('/'), sensor_name + '#0'):
            try:
                with self.lock:
                    self.sensor = self.sim.getObject(cand)
                self.sensor_name = cand
                break
            except Exception:
                pass
        if self.sensor is None:
            raise RuntimeError(f"[vision_web] cannot find vision sensor '{sensor_name}'")

        # Resolve mat alias (optional)
        self.mat = None
        try:
            with self.lock:
                self.mat = self.sim.getObject(mat_alias)
        except Exception:
            print(f"[vision_web] WARN: could not resolve mat alias '{mat_alias}'. Overlay will assume mat at origin.")
            self.mat = None

        # Explicit handling?
        self.explicit = False
        try:
            with self.lock:
                self.explicit = bool(self.sim.getObjectInt32Param(
                    self.sensor, self.sim.visionintparam_explicit_handling))
        except Exception:
            pass
        print(f"[vision_web] sensor='{self.sensor_name}', explicitHandling={self.explicit}, mat='{self.mat_alias}', Q={self.jpg_q}, scale={self.scale}")

        # Ensure simulation running
        try:
            with self.lock:
                st = self.sim.getSimulationState()
                if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                    print('[vision_web] starting simulation...')
                    self.sim.startSimulation()
        except Exception as e:
            print('[vision_web] WARN cannot start simulation:', e)

        # Cache perspective mode & angle (updated lazily too)
        self.perspective = True
        self.vfov = None  # radians
        try:
            with self.lock:
                self.perspective = bool(self.sim.getObjectInt32Param(self.sensor, self.sim.visionintparam_perspective_mode))
            if self.perspective:
                with self.lock:
                    self.vfov = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_perspective_angle))
        except Exception:
            pass

        # Warm-up
        for _ in range(5):
            self._handle_if_needed()
            self._step()
            time.sleep(0.01)

    # ---- sim helpers ----
    def _step(self):
        try:
            with self.lock:
                self.sim.step()
        except Exception:
            pass

    def _handle_if_needed(self):
        if self.explicit:
            try:
                with self.lock:
                    self.sim.handleVisionSensor(self.sensor)
            except Exception as e:
                print('[vision_web] handleVisionSensor failed:', e)

    # ---- image IO ----
    def _read_raw_image(self):
        """
        Return (arr,resX,resY). arr is 1D uint8 (len=resX*resY*(1|3|4)).
        Correct unpacking for ZeroMQ API:
          sim.getVisionSensorImg(...) -> (image_bytes, [resX,resY])
        """
        # Preferred API (bytes + [w,h])
        try:
            with self.lock:
                img, res = self.sim.getVisionSensorImg(self.sensor)
            resX, resY = int(res[0]), int(res[1])
        except Exception:
            # Fallback: char image (bytes, w, h)
            try:
                with self.lock:
                    img, resX, resY = self.sim.getVisionSensorCharImage(self.sensor)
                resX, resY = int(resX), int(resY)
            except Exception as e1:
                # Last resort: float image
                try:
                    with self.lock:
                        out = self.sim.getVisionSensorImage(self.sensor)
                    if isinstance(out, (list, tuple)) and len(out) == 3:
                        img, resX, resY = out
                        resX, resY = int(resX), int(resY)
                    else:
                        img, res = out
                        resX, resY = int(res[0]), int(res[1])
                    img = (np.array(img, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                    return img, resX, resY
                except Exception as e2:
                    raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")

        # Normalize to uint8 numpy
        if isinstance(img, (bytes, bytearray)):
            arr = np.frombuffer(img, dtype=np.uint8)
        else:
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = (arr.astype(np.float32).clip(0, 255)).astype(np.uint8)
        return arr, resX, resY

    def _frame_from_arr(self, arr, w, h):
        n = arr.size
        if n == w * h:            # Gray
            frame = arr.reshape(h, w)
            frame = np.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif n == w * h * 3:      # RGB
            frame = arr.reshape(h, w, 3)
            frame = np.flip(frame, 0)[:, :, ::-1]  # RGB->BGR
        elif n == w * h * 4:      # RGBA
            frame = arr.reshape(h, w, 4)
            frame = np.flip(frame, 0)[:, :, :3][:, :, ::-1]  # drop A, RGB->BGR
        else:
            # Try to infer channels if divisible
            if n % (w * h) == 0:
                c = n // (w * h)
                frame = arr.reshape(h, w, c)
                frame = np.flip(frame, 0)
                if c == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif c >= 3:
                    frame = frame[:, :, :3][:, :, ::-1]
            else:
                raise RuntimeError(f"unexpected buffer size: n={n}, w={w}, h={h}")
        return frame

    def grab_raw_frame(self):
        """Return raw (or server-upscaled) BGR frame (H,W,3)."""
        for _ in range(10):
            self._handle_if_needed()
            arr, w, h = self._read_raw_image()
            if w > 0 and h > 0 and arr.size >= w * h:
                frame = self._frame_from_arr(arr, w, h)
                self.last_res = (w, h)
                if self.scale != 1.0:
                    new_w = max(1, int(round(frame.shape[1] * self.scale)))
                    new_h = max(1, int(round(frame.shape[0] * self.scale)))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                return frame
            self._step()
            time.sleep(0.01)
        raise RuntimeError(f"vision sensor not ready (w={w}, h={h}, n={arr.size})")

    # ---- projection helpers ----
    @staticmethod
    def _M12_to_T(m12):
        T = np.eye(4, dtype=float)
        T[:3, :4] = np.array(m12, dtype=float).reshape(3, 4)
        return T

    def _get_T_cam_mat(self):
        if self.mat is None:
            return np.eye(4, dtype=float)
        with self.lock:
            M = self.sim.getObjectMatrix(self.sensor, self.mat)  # mat -> cam
        return self._M12_to_T(M)

    def _get_intrinsics(self, w, h):
        if self.perspective and (self.vfov is None or not np.isfinite(self.vfov)):
            try:
                with self.lock:
                    self.vfov = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_perspective_angle))
            except Exception:
                self.vfov = math.radians(60.0)
        cx, cy = 0.5 * w, 0.5 * h
        if self.perspective:
            fy = (h * 0.5) / math.tan(0.5 * (self.vfov or math.radians(60.0)))
            fx = fy * (w / h)
            return fx, fy, cx, cy, True
        else:
            try:
                with self.lock:
                    ortho_size = float(self.sim.getObjectFloatParam(self.sensor, self.sim.visionfloatparam_ortho_size))
            except Exception:
                ortho_size = 0.2
            m_per_pix_y = (2.0 * ortho_size) / h
            m_per_pix_x = m_per_pix_y
            return m_per_pix_x, m_per_pix_y, cx, cy, False

    def _project_pts(self, P_cam: np.ndarray, w: int, h: int):
        fx, fy, cx, cy, is_persp = self._get_intrinsics(w, h)
        uv = []
        for x, y, z in P_cam:
            if is_persp:
                if z <= 1e-6:
                    uv.append([np.nan, np.nan]); continue
                u = fx * (x / z) + cx
                v = -fy * (y / z) + cy
            else:
                u = (x / (2.0 * fx)) * w + cx
                v = (-y / (2.0 * fy)) * h + cy
            uv.append([u, v])
        return np.array(uv)

    def _overlay_paths(self, frame, T_cam_mat, cuts_dict, color_cut=(60, 220, 60), color_bite=(255, 180, 60)):
        H, W = frame.shape[:2]
        spacing = float(cuts_dict.get('params', {}).get('spacing', 0.006))
        bite    = float(cuts_dict.get('params', {}).get('bite',    0.005))

        # thickness roughly scales with resolution
        thick = max(1, int(round(min(W, H) / 400)))

        for cut in cuts_dict.get('cuts', []):
            poly = np.array(cut.get('polyline', []), dtype=float)  # Nx2 or Nx3
            if poly.size == 0:
                continue
            poly_xy = poly[:, :2] if poly.shape[1] >= 2 else poly

            pts, entries, exits = plan_bites_2d(poly_xy, spacing, bite)

            def to3(p2): return np.c_[p2, np.zeros((len(p2), 1))]
            P_cut    = to3(poly_xy)
            P_entry  = to3(entries) if len(entries) else np.empty((0, 3))
            P_exit   = to3(exits)   if len(exits)   else np.empty((0, 3))

            def xform(P):
                if P.size == 0: return P
                P_h = np.c_[P, np.ones((len(P), 1))]
                return (T_cam_mat @ P_h.T).T[:, :3]

            C_cut   = xform(P_cut)
            C_entry = xform(P_entry)
            C_exit  = xform(P_exit)

            U_cut   = self._project_pts(C_cut,  W, H)
            U_entry = self._project_pts(C_entry,W, H)
            U_exit  = self._project_pts(C_exit, W, H)

            pts_int = []
            for u, v in U_cut:
                if np.isfinite(u) and np.isfinite(v):
                    pts_int.append((int(round(u)), int(round(v))))
            if len(pts_int) >= 2:
                cv2.polylines(frame, [np.array(pts_int, dtype=np.int32)], False, color_cut, thick, cv2.LINE_AA)

            for (ue, ve), (ux, vx) in zip(U_entry, U_exit):
                if not (np.isfinite(ue) and np.isfinite(ve) and np.isfinite(ux) and np.isfinite(vx)):
                    continue
                p1 = (int(round(ue)), int(round(ve)))
                p2 = (int(round(ux)), int(round(vx)))
                cv2.line(frame, p1, p2, color_bite, thick, cv2.LINE_AA)
                cv2.circle(frame, p1, max(2, thick+1), (0, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p2, max(2, thick+1), (0, 128, 255), -1, cv2.LINE_AA)

    # ---- generators & snapshots ----
    def mjpeg_generator(self, overlay=True):
        delay = 1.0 / self.fps
        while True:
            try:
                self._step()
                frame = self.grab_raw_frame()
                if overlay:
                    cuts = self.cuts_buf.get() if self.cuts_buf else None
                    T = self._get_T_cam_mat()
                    if cuts is not None:
                        try:
                            self._overlay_paths(frame, T, cuts)
                        except Exception as e:
                            cv2.putText(frame, f'Overlay error: {e}', (10, 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 240), 2, cv2.LINE_AA)
                ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_q])
                if not ok:
                    time.sleep(delay); continue
                chunk = jpg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Cache-Control: no-cache\r\n'
                       b'Pragma: no-cache\r\n'
                       b'Content-Length: ' + str(len(chunk)).encode() + b'\r\n\r\n' +
                       chunk + b'\r\n')
            except Exception as e:
                print('[vision_web] MJPEG error:', e)
                time.sleep(0.25)

    def snapshot_bytes(self, overlay=True):
        self._step()
        frame = self.grab_raw_frame()
        if overlay:
            cuts = self.cuts_buf.get() if self.cuts_buf else None
            T = self._get_T_cam_mat()
            if cuts is not None:
                self._overlay_paths(frame, T, cuts)
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(self.jpg_q, 92)])
        if not ok:
            raise RuntimeError('encode failed')
        return jpg.tobytes()


# -------------- File helpers --------------
def find_models_root():
    # 1) installed share/suture_arm/ml
    if get_package_share_directory:
        try:
            share = get_package_share_directory('suture_arm')
            p = os.path.join(share, 'ml')
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    # 2) dev tree fallbacks
    here = os.path.dirname(__file__)
    for cand in ('ML_detection', os.path.join('..', 'ML_detection')):
        p = os.path.abspath(os.path.join(here, cand))
        if os.path.isdir(p):
            return p
    # 3) cwd
    return os.getcwd()


# -------------- Flask app factory --------------
def create_app(sensor_alias='/visionSensor', fps=10, mat_alias='/mat', jpg_quality=92, server_scale=1.0):
    cuts_buf = CutsBuffer()
    start_cuts_listener_in_thread(cuts_buf)

    vs = VisionStreamer(sensor_alias, fps, mat_alias, cuts_buf, jpg_quality, server_scale)
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(INDEX_HTML, sensor=vs.sensor_name, fps=fps, mat=mat_alias,
                                      quality=jpg_quality, scale=server_scale)

    @app.route('/video')
    def video():
        resp = Response(vs.mjpeg_generator(overlay=True),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Connection'] = 'close'
        return resp

    @app.route('/video_raw')
    def video_raw():
        resp = Response(vs.mjpeg_generator(overlay=False),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Connection'] = 'close'
        return resp

    @app.route('/snapshot')
    def snapshot():
        try:
            data = vs.snapshot_bytes(overlay=True)
            resp = make_response(data)
            resp.headers['Content-Type'] = 'image/jpeg'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        except Exception as e:
            tb = traceback.format_exc()
            return make_response(f"SNAPSHOT ERROR:\n{e}\n\n{tb}", 500)

    @app.route('/snapshot_raw')
    def snapshot_raw():
        try:
            data = vs.snapshot_bytes(overlay=False)
            resp = make_response(data)
            resp.headers['Content-Type'] = 'image/jpeg'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        except Exception as e:
            tb = traceback.format_exc()
            return make_response(f"SNAPSHOT RAW ERROR:\n{e}\n\n{tb}", 500)

    @app.route('/models')
    def models():
        root = find_models_root()
        files = []
        for f in sorted(os.listdir(root)):
            if f.lower().endswith('.pth'):
                fp = os.path.join(root, f)
                try:
                    size = os.path.getsize(fp)
                except Exception:
                    size = 0
                files.append((f, sizeof_fmt(size)))
        return render_template_string(MODELS_HTML, root=root, files=files)

    @app.route('/models/<path:fname>')
    def download_model(fname):
        root = find_models_root()
        return send_from_directory(root, fname, as_attachment=True)

    @app.route('/cuts')
    def cuts_debug():
        return jsonify(cuts_buf.get() or {})

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
        })

    return app


# -------------- entry point --------------
def main():
    ap = argparse.ArgumentParser(description='Stream CoppeliaSim visionSensor to browser with suture overlay')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--sensor', default='/visionSensor')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--mat', default='/mat', help='Alias/path of the dummy that defines the mat frame')
    ap.add_argument('--quality', type=int, default=92, help='JPEG quality (60..100)')
    ap.add_argument('--server_scale', type=float, default=1.0, help='Resize factor before encoding (1.0 = off)')
    args = ap.parse_args()

    app = create_app(args.sensor, args.fps, args.mat, args.quality, args.server_scale)
    print(f"[vision_web] Serving on http://{args.host}:{args.port}/  (sensor={args.sensor}, mat={args.mat}, Q={args.quality}, scale={args.server_scale}x)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)  # thread-safe due to internal lock


if __name__ == '__main__':
    main()
