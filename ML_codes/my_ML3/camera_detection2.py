#!/usr/bin/env python3
"""
RealSense/Webcam → Mask R-CNN inference → Flask two-stream:
 - /masks : image with mask overlays
 - /stitch: image with continuous zig-zag stitch (insertion points = circles)

Requirements:
 pip install flask opencv-python torch torchvision pycocotools pyrealsense2

Place your Mask R-CNN state_dict at: mask_cuts_detector.pth
(If you saved full model object instead, adapt loading accordingly.)
"""

import os
import time
import threading
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ----------- Config -----------
WEIGHTS_PATH = "mask_cuts_detector_best.pth"   # expected to be model.state_dict()
CONF_THRESH = 0.5
MAX_MASKS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAME_W = 640
FRAME_H = 480
STITCH_OFFSET_PX = 10
STITCH_SPACING_MIN = 8
STITCH_SAMPLE_POINTS = 24
JPEG_QUALITY = 80
# ------------------------------

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head><title>Live Wound Masks & Stitch</title>
<style>
  body { font-family: Arial, sans-serif; text-align:center; background:#f7f7f7; }
  .row { display:flex; justify-content:center; gap:20px; margin-top:18px; }
  .panel { background: #fff; padding:10px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
  img { width: 640px; height: 480px; border:1px solid #222; }
  h1 { margin-top:14px; }
</style>
</head>
<body>
  <h1>Live Wound Masks (left) & Stitch Path (right)</h1>
  <div class="row">
    <div class="panel"><h3>Masks</h3><img src="{{ url_for('masks_feed') }}"></div>
    <div class="panel"><h3>Stitch</h3><img src="{{ url_for('stitch_feed') }}"></div>
  </div>
  <p style="margin-top:12px">Press Ctrl+C in terminal to stop.</p>
</body>
</html>
"""

# ---------- Model building & loading ----------
def build_maskrcnn(num_classes: int = 2):
    # instantiate base model without pretrained heads
    model = maskrcnn_resnet50_fpn(weights=None)
    # replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_weights_to_model(model, weights_path: str, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    # If a dict nested under key, try to extract common keys
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # keep as-is; load_state_dict will handle
        pass
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ---------- Camera + inference background thread ----------
# Shared variables updated by capture thread
_latest_frame = None          # BGR numpy array (raw frame)
_latest_masks = []            # List[np.ndarray] boolean masks (H x W)
_latest_scores = []           # List[float]
_lock = threading.Lock()
_run_capture = True

# Try RealSense, else webcam
try:
    import pyrealsense2 as rs
    _has_realsense = True
except Exception:
    _has_realsense = False

def capture_loop(model, device):
    global _latest_frame, _latest_masks, _latest_scores, _run_capture

    # Setup camera
    if _has_realsense:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        pipeline.start(cfg)
        print("Using RealSense camera.")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam and RealSense not available.")
        print("RealSense not available — using default webcam.")

    transform = lambda img_bgr: F.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(device)

    try:
        while _run_capture:
            if _has_realsense:
                frames = pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            # small safety resize (already requested), but ensure correct shape
            frame_proc = cv2.resize(frame, (FRAME_W, FRAME_H))

            # Inference: single image
            img_tensor = transform(frame_proc).unsqueeze(0)  # [1,C,H,W]
            with torch.no_grad():
                outputs = model(img_tensor)[0]

            scores = outputs.get("scores", torch.tensor([])).cpu().numpy()
            masks_raw = outputs.get("masks", None)

            masks_out = []
            scores_out = []
            if masks_raw is not None:
                nmasks = min(masks_raw.shape[0], len(scores))
                for i in range(nmasks):
                    if scores[i] >= CONF_THRESH:
                        mask_bool = (masks_raw[i, 0].cpu().numpy() > 0.5)
                        masks_out.append(mask_bool)
                        scores_out.append(float(scores[i]))
                        if len(masks_out) >= MAX_MASKS:
                            break

            with _lock:
                _latest_frame = frame_proc.copy()
                _latest_masks = masks_out
                _latest_scores = scores_out

            # small delay to avoid busy loop (control FPS)
            time.sleep(0.01)
    finally:
        _run_capture = False
        if _has_realsense:
            try:
                pipeline.stop()
            except Exception:
                pass
        else:
            try:
                cap.release()
            except Exception:
                pass

# ---------- Utilities for masks & stitching ----------
def mask_to_contours(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def overlay_mask(image: np.ndarray, mask_bool: np.ndarray, color=(0,200,0), alpha=0.45):
    colored = np.zeros_like(image, dtype=np.uint8)
    colored[mask_bool] = color
    out = cv2.addWeighted(image, 1.0, colored, alpha, 0)
    # draw outline
    mask_u = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)
    contours = mask_to_contours(mask_u)
    cv2.drawContours(out, contours, -1, (0,160,0), 2)
    return out

def compute_principal_axis(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    pts_centered = pts - mean
    if pts_centered.shape[0] < 2:
        return np.array([1.0, 0.0]), mean
    _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
    axis = vh[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis, mean

def sample_midpoints_from_mask(mask_bool: np.ndarray, n_points: int = STITCH_SAMPLE_POINTS) -> List[Tuple[int,int]]:
    h, w = mask_bool.shape
    mask_u8 = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)
    contours = mask_to_contours(mask_u8)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 16:
        return []
    axis, centroid = compute_principal_axis(contour)
    perp = np.array([-axis[1], axis[0]])

    pts = contour.reshape(-1, 2).astype(np.float32)
    projections = ((pts - centroid) @ axis)
    min_p, max_p = projections.min(), projections.max()
    if max_p - min_p < 1e-3:
        return []

    t_vals = np.linspace(min_p, max_p, n_points)
    midpoints = []
    max_search = int(max(h,w) * 0.6)
    for t in t_vals:
        probe = centroid + axis * t
        line_coords = []
        for s in np.linspace(-max_search/2, max_search/2, max_search):
            p = probe + perp * s
            ix = int(round(p[0])); iy = int(round(p[1]))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if mask_bool[iy, ix]:
                line_coords.append((ix, iy))
        if not line_coords:
            continue
        xs = [c[0] for c in line_coords]; ys = [c[1] for c in line_coords]
        mid_x = int(round(np.mean(xs))); mid_y = int(round(np.mean(ys)))
        midpoints.append((mid_x, mid_y))

    if not midpoints:
        return []
    # filter by spacing
    filtered = [midpoints[0]]
    for p in midpoints[1:]:
        last = filtered[-1]
        if (p[0]-last[0])**2 + (p[1]-last[1])**2 >= STITCH_SPACING_MIN**2:
            filtered.append(p)
    return filtered

def stitch_points_from_midline(midpoints: List[Tuple[int,int]], offset_px=STITCH_OFFSET_PX):
    if len(midpoints) < 2:
        return [], []
    pts = np.array(midpoints, dtype=np.float32)
    vectors = np.diff(pts, axis=0)
    vectors = np.vstack([vectors, vectors[-1]])
    stitch_pts = []
    path = []
    for i, p in enumerate(pts):
        v = vectors[i]
        norm = np.linalg.norm(v)
        if norm < 1e-3:
            perp = np.array([0.0, 1.0])
        else:
            unit = v / norm
            perp = np.array([-unit[1], unit[0]])
        direction = 1 if (i % 2 == 0) else -1
        spt = (int(round(p[0] + direction * offset_px * perp[0])),
               int(round(p[1] + direction * offset_px * perp[1])))
        stitch_pts.append(spt)
        path.append(spt)
    return stitch_pts, path

# ---------- Frame annotators used by Flask generators ----------
def annotate_masks_view(frame: np.ndarray, masks: List[np.ndarray]):
    out = frame.copy()
    for mask in masks:
        out = overlay_mask(out, mask, color=(0,200,0), alpha=0.45)
    return out

def annotate_stitch_view(frame: np.ndarray, masks: List[np.ndarray]):
    out = frame.copy()
    for mask in masks:
        midpoints = sample_midpoints_from_mask(mask, n_points=STITCH_SAMPLE_POINTS)
        if len(midpoints) < 2:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            mid_x = (x1 + x2)//2
            midpoints = [(mid_x, y) for y in range(y1+5, y2-5, max(8,(y2-y1)//6))]
        stitch_pts, path = stitch_points_from_midline(midpoints, offset_px=STITCH_OFFSET_PX)
        # draw circles
        for p in stitch_pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0,0,255), -1)  # red insertion circle
        # draw connecting thread
        if len(path) >= 2:
            for i in range(len(path)-1):
                cv2.line(out, path[i], path[i+1], (255,0,0), 2)
    return out

# ---------- Flask frame generators that read shared latest frame ----------
def frame_to_jpeg_bytes(frame: np.ndarray):
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ret:
        return None
    return buf.tobytes()

def gen_masks_feed():
    while True:
        with _lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
            masks = list(_latest_masks)
        if frame is None:
            time.sleep(0.01)
            continue
        out = annotate_masks_view(frame, masks)
        b = frame_to_jpeg_bytes(out)
        if b is None:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b + b'\r\n')

def gen_stitch_feed():
    while True:
        with _lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
            masks = list(_latest_masks)
        if frame is None:
            time.sleep(0.01)
            continue
        out = annotate_stitch_view(frame, masks)
        b = frame_to_jpeg_bytes(out)
        if b is None:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b + b'\r\n')

# ---------- Flask routes ----------
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/masks')
def masks_feed():
    return Response(gen_masks_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stitch')
def stitch_feed():
    return Response(gen_stitch_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- Main ----------
def main():
    global _run_capture
    # Build and load model
    print("Device:", DEVICE)
    model = build_maskrcnn(num_classes=2)
    try:
        load_weights_to_model(model, WEIGHTS_PATH, DEVICE)
    except Exception as e:
        raise SystemExit(f"Failed to load weights: {e}")

    # start capture thread
    t = threading.Thread(target=capture_loop, args=(model, DEVICE), daemon=True)
    t.start()

    try:
        print("Server running at http://0.0.0.0:5000 — open in browser")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        _run_capture = False
        t.join(timeout=1)
        print("Shutting down.")

if __name__ == "__main__":
    main()
