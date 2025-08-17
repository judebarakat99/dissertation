#!/usr/bin/env python3
import os
import io
import time
import argparse
import numpy as np

from flask import Flask, Response, send_from_directory, render_template_string

# ZMQ Remote API (CoppeliaSim)
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback

# JPEG encoding (install: pip install opencv-python)
import cv2

# ROS/ament share lookup (for installed package data)
try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>UR3 Vision — Suture Pad View</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;}
    header{display:flex;gap:16px;align-items:center;}
    img{max-width:100%;border:1px solid #ddd;border-radius:8px;}
    .meta{color:#666;font-size:0.9rem}
    nav a{margin-right:12px}
  </style>
</head>
<body>
  <header>
    <h2>UR3 Vision — Top view</h2>
    <nav>
      <a href="/">Live</a>
      <a href="/models">Model files</a>
    </nav>
  </header>
  <div class="meta">Source: visionSensor (CoppeliaSim) • MJPEG stream • {{fps}} FPS</div>
  <p><img src="/video" alt="vision stream"/></p>
</body>
</html>
"""

MODELS_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Detector model files</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;}
    table{border-collapse:collapse;min-width:420px}
    td,th{border:1px solid #ddd;padding:8px;}
    th{background:#f7f7f7;text-align:left}
    code{background:#f2f2f2;padding:2px 4px;border-radius:4px}
  </style>
</head>
<body>
  <h2>Detector model files</h2>
  <p>Directory: <code>{{root}}</code></p>
  {% if files %}
  <table>
    <tr><th>File</th><th>Size</th><th>Download</th></tr>
    {% for f,s in files %}
      <tr>
        <td>{{f}}</td>
        <td>{{s}}</td>
        <td><a href="/models/{{f}}">download</a></td>
      </tr>
    {% endfor %}
  </table>
  {% else %}
    <p><em>No *.pth files found here.</em></p>
  {% endif %}
  <p><a href="/">⟵ Back to live view</a></p>
</body>
</html>
"""


def sizeof_fmt(num):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


class VisionStreamer:
    def __init__(self, sensor_name='/visionSensor', fps=10):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.fps = fps

        # step the sim deterministically (works with the ZMQ add-on)
        if hasattr(self.sim, 'setStepping'):
            self.sim.setStepping(True)

        # find the sensor (try a few aliases)
        self.sensor = None
        for cand in (sensor_name, sensor_name.lstrip('/'), sensor_name + '#0'):
            try:
                self.sensor = self.sim.getObject(cand)
                break
            except Exception:
                pass
        if self.sensor is None:
            raise RuntimeError(f"Cannot find vision sensor '{sensor_name}'")

        # ensure simulation is running
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

    def _grab_frame(self):
        """
        Robustly fetch an RGB frame from CoppeliaSim and return a numpy uint8 (H,W,3) BGR image.
        Tries several API signatures to be compatible across versions.
        """
        img = None
        resX = resY = None
        # Attempt 1: common signature
        try:
            img, resX, resY = self.sim.getVisionSensorImg(self.sensor)
        except Exception:
            # Attempt 2: some versions return (res, img)
            try:
                res, img = self.sim.getVisionSensorImg(self.sensor)
                resX, resY = int(res[0]), int(res[1])
            except Exception:
                # Attempt 3: char image variant
                img, resX, resY = self.sim.getVisionSensorCharImage(self.sensor)

        # Convert to numpy
        if isinstance(img, (bytes, bytearray)):
            arr = np.frombuffer(img, dtype=np.uint8)
        else:
            arr = np.array(img, dtype=np.uint8)

        # Some APIs return floats 0..255; coerce
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # Infer resolution if not provided (square guess avoided; better to error)
        if resX is None or resY is None:
            # Try to infer from length — assume 3 channels
            n = arr.size
            s = int(round((n/3)**0.5))
            if s*s*3 != n:
                raise RuntimeError("Unable to infer image resolution from sensor output.")
            resX = resY = s

        try:
            frame = arr.reshape(resY, resX, 3)
        except Exception as e:
            raise RuntimeError(f"Failed to reshape image buffer: {e}")

        # Coppelia images are vertically flipped; also convert RGB->BGR for OpenCV
        frame = np.flip(frame, 0)[:, :, ::-1]
        return frame

    def mjpeg_generator(self):
        delay = 1.0 / max(1, self.fps)
        while True:
            try:
                frame = self._grab_frame()
                ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    time.sleep(delay)
                    continue
                chunk = jpg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(chunk)).encode() + b'\r\n\r\n' +
                       chunk + b'\r\n')
            except Exception as e:
                # Back-off on error (sensor not ready, etc.)
                time.sleep(0.2)
            # advance sim one step if stepping is enabled
            try:
                self.sim.step()
            except Exception:
                pass
            time.sleep(delay)


def find_models_root():
    """
    Where are the *.pth files?
    1) Prefer installed share/suture_arm/ml
    2) Fall back to package source tree ML_detection (during dev)
    """
    # 1) installed
    if get_package_share_directory:
        try:
            share = get_package_share_directory('suture_arm')
            p = os.path.join(share, 'ml')
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    # 2) dev tree
    here = os.path.dirname(__file__)
    for cand in ('ML_detection', os.path.join('..', 'ML_detection')):
        p = os.path.abspath(os.path.join(here, cand))
        if os.path.isdir(p):
            return p
    # last resort: current dir
    return os.getcwd()


def create_app(sensor_alias='/visionSensor', fps=10):
    vs = VisionStreamer(sensor_alias, fps)
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(INDEX_HTML, fps=fps)

    @app.route('/video')
    def video():
        return Response(vs.mjpeg_generator(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

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

    return app


def main():
    parser = argparse.ArgumentParser(description='Stream visionSensor to a browser, and list model files.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--sensor', default='/visionSensor', help='Object alias/path of vision sensor')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    app = create_app(args.sensor, args.fps)
    # threaded=True keeps MJPEG streaming responsive
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
