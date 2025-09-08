#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import io, os, time, json, threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from flask import Flask, Response, render_template, abort, make_response, request, redirect, url_for, jsonify
from ament_index_python.packages import get_package_share_directory

# ---- local modules (you created these) ----
# csim_frames MUST export: FrameGrabber, mjpeg_generator_raw, encode_jpeg
from .csim_frames import FrameGrabber, mjpeg_generator_raw, encode_jpeg

# model_interface MUST export: TORCH_OK, _TORCH_ERR, load_detector, run_inference, draw_dets
from .model_interface import TORCH_OK, _TORCH_ERR, load_detector, run_inference, draw_dets

# image_to_map_mapper MUST export: load_pad_mapper(calib_path:str)->dict|None, get_px_per_mm(...)
from .image_to_map_mapper import load_pad_mapper, get_px_per_mm

# ---- stitching (pattern drawing used by pattern.jpg + path gen) ----
try:
    from . import stitching
except Exception:
    import stitching


# ====================== Config ======================

CSIM_HOST = os.getenv("CSIM_HOST", "127.0.0.1")
CSIM_PORT = int(os.getenv("CSIM_PORT", "23000"))
TOP_SENSOR_ALIAS  = os.getenv("TOP_SENSOR_ALIAS", "/visionSensor")
SIDE_SENSOR_ALIAS = os.getenv("SIDE_SENSOR_ALIAS", "/visionSensor_SideView")

FPS = int(os.getenv("VISION_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))

STEPPED_SNAPSHOT = int(os.getenv("STEPPED_SNAPSHOT", "1"))
STEPPED_RAW      = int(os.getenv("STEPPED_RAW", "0"))

CALIB_PATH = os.getenv("SUTURE_PAD_CALIB", "")  # may be "" (that’s fine)

# ====================== Templates & Models ======================

def _find_templates_dir() -> str:
    env = os.getenv("SUTURE_ARM_TEMPLATES", "").strip()
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
    env = os.getenv("SUTURE_ARM_ML", "").strip()
    if env and os.path.isdir(env): return env
    try:
        share = get_package_share_directory("suture_arm")
        mdir = os.path.join(share, "ml")
        if os.path.isdir(mdir): return mdir
    except Exception:
        pass
    here = os.path.dirname(__file__)
    cand = os.path.abspath(os.path.join(here, "..", "ML_detection"))
    return cand if os.path.isdir(cand) else here

TEMPLATES_DIR = _find_templates_dir()
MODELS_ROOT   = _find_models_root()

if not os.path.isfile(os.path.join(TEMPLATES_DIR, "index.html")):
    raise RuntimeError(f"index.html not found in {TEMPLATES_DIR}. Put it in package share or set SUTURE_ARM_TEMPLATES.")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ====================== Optional calibration ======================

PAD_MAPPER = None  # loaded on startup

# ====================== ROS2 persistent publisher ======================

cuts_node = None
cuts_pub  = None
targets_pub = None
def init_ros2_publisher():
    global cuts_node, cuts_pub, targets_pub
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
                # Publish both the original suture cuts and vision targets.
                self.pub_cuts = self.create_publisher(_MsgString, '/suture_cuts', 10)
                self.pub_targets = self.create_publisher(_MsgString, '/vision_targets', 10)

        cuts_node = _CutsNode()
        cuts_pub  = cuts_node.pub_cuts
        targets_pub = cuts_node.pub_targets
        print("[vision_web] ROS2 publishers ready on /suture_cuts and /vision_targets", flush=True)

        def _spin_trickle():
            while rclpy.ok():
                try:
                    rclpy.spin_once(cuts_node, timeout_sec=0.05)
                except Exception:
                    break
                time.sleep(0.2)
        threading.Thread(target=_spin_trickle, daemon=True).start()

    except Exception as e:
        cuts_node, cuts_pub = None, None
        print(f"[vision_web][ERR] ROS2 publisher init failed: {e}", flush=True)

# ====================== Grabbing (via csim_frames) ======================

grabber = FrameGrabber(CSIM_HOST, CSIM_PORT, TOP_SENSOR_ALIAS, stepped=STEPPED_SNAPSHOT, fps=FPS)

# ====================== Detection state ======================

def _available_models() -> List[str]:
    try:
        files = [f for f in os.listdir(MODELS_ROOT) if f.lower().endswith(".pth")]
        files.sort(key=lambda n: (0 if "cuts_maskrcnn_best" in n.lower() else 1, n.lower()))
        return files
    except Exception:
        return []

def _select_weights_file() -> Optional[str]:
    models = _available_models()
    return os.path.join(MODELS_ROOT, models[0]) if models else None

DETECTOR      = None  # (model, class_names, device)
WEIGHTS_PATH  = None
LAST_DETS     : Optional[Dict] = None
LAST_FRAME_PIL: Optional["Image.Image"] = None
SELECTED_DET_ID: Optional[int] = None
LAST_MASKS    : Optional[List[np.ndarray]] = None

# ====================== Helpers ======================

def _get_selected_crop_and_mask_np(det_id: Optional[int], pad: float = 0.08, outw: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
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

    # asymmetric pad (wider horizontally)
    pad_w = float(request.args.get("pad_w", default=pad * 2.2))
    pad_h = float(request.args.get("pad_h", default=pad))
    pw = pad_w * (x2 - x1); ph = pad_h * (y2 - y1)

    xx1 = max(0, int(np.floor(x1 - pw))) - 10
    yy1 = max(0, int(np.floor(y1 - ph)))
    xx2 = 10 + min(int(np.ceil(x2 + pw)), int(W))
    yy2 = min(int(np.ceil(y2 + ph)), int(H))

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

# --- NEW helper: crop + mask + meta (offsets & full size) ---
def _get_selected_crop_mask_and_meta(det_id: Optional[int], pad: float = 0.08, outw: Optional[int] = None):
    """
    Return:
      crop_bgr, crop_mask  - arrays in crop coordinates
      xx1, yy1             - crop top-left in FULL image pixel coordinates
      W_full, H_full       - full image dimensions in pixels
    """
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
    W_full, H_full = LAST_DETS.get("image_size", [LAST_FRAME_PIL.width, LAST_FRAME_PIL.height])

    # asymmetric pad (wider horizontally); mirrors your snapshot route defaults
    pad_w = float(request.args.get("pad_w", default=pad * 2.2))
    pad_h = float(request.args.get("pad_h", default=pad))
    pw = pad_w * (x2 - x1); ph = pad_h * (y2 - y1)

    xx1 = max(0, int(np.floor(x1 - pw))) - 10
    yy1 = max(0, int(np.floor(y1 - ph)))
    xx2 = 10 + min(int(np.ceil(x2 + pw)), int(W_full))
    yy2 = min(int(np.ceil(y2 + ph)), int(H_full))

    crop_pil = LAST_FRAME_PIL.crop((xx1, yy1, xx2, yy2))
    crop_rgb = np.array(crop_pil)
    if crop_rgb.ndim == 2:
        crop_rgb = np.repeat(crop_rgb[..., None], 3, axis=2)
    crop_bgr = crop_rgb[:, :, ::-1].copy()

    full_mask = LAST_MASKS[det_id]  # HxW uint8 in FULL image coords
    crop_mask = full_mask[yy1:yy2, xx1:xx2].copy()

    if outw and outw > 0:
        w, h = crop_pil.size
        outh = max(1, int(round(h * (outw / float(w)))))
        crop_bgr = cv2.resize(crop_bgr, (outw, outh), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(crop_mask, (outw, outh), interpolation=cv2.INTER_NEAREST)

    return crop_bgr, crop_mask, int(xx1), int(yy1), int(W_full), int(H_full)


def _hex_to_rgb(s: str, default=(0,220,220)):
    try:
        s = (s or "").strip()
        if s.startswith("#"): s = s[1:]
        if len(s) == 3: s = ''.join([c*2 for c in s])
        if len(s) != 6: return default
        r = int(s[0:2],16); g = int(s[2:4],16); b = int(s[4:6],16)
        return (r,g,b)
    except Exception:
        return default

def _hex_to_bgr(s: str, default=(30,200,255)):
    r,g,b = _hex_to_rgb(s, default=(default[2], default[1], default[0]))
    return (b,g,r)

def _color_to_scalar(c) -> int:
    """Convert BGR/RGB tuple/list to a single scalar 0..255 (grayscale-ish).
    If already a number, clamp to 0..255 and return int.
    """
    try:
        if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 3:
            return int(max(0, min(255, 0.114 * float(c[0]) + 0.587 * float(c[1]) + 0.299 * float(c[2]))))
        return int(max(0, min(255, float(c))))
    except Exception:
        return 200  # safe fallback brightness

# ====================== Routes ======================

@app.route("/")
def index():
    model_name = os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else "(none)"
    return render_template(
        "index.html",
        top_sensor=TOP_SENSOR_ALIAS,
        side_sensor=SIDE_SENSOR_ALIAS,
        host=CSIM_HOST, port=CSIM_PORT,
        model=model_name, models=_available_models(),
    )

#====================== Select model ======================
@app.route("/select_model", methods=["POST"])
def select_model():
    fname = request.form.get("weights", "").strip()
    if not fname: return redirect(url_for("index"))
    candidate = os.path.join(MODELS_ROOT, fname)
    if not os.path.isfile(candidate): return redirect(url_for("index"))
    if not TORCH_OK: return redirect(url_for("index"))
    global DETECTOR, WEIGHTS_PATH, LAST_DETS, LAST_FRAME_PIL, LAST_MASKS, SELECTED_DET_ID
    try:
        DETECTOR = load_detector(candidate)
        WEIGHTS_PATH = candidate
        LAST_DETS = LAST_FRAME_PIL = LAST_MASKS = None
        SELECTED_DET_ID = None
    except Exception as e:
        print(f"[vision_web][ERR] failed to load {candidate}: {e}", flush=True)
    return redirect(url_for("index"))

#====================== Raw stream ======================
@app.route("/stream_raw/<which>")
def stream_raw(which: str):
    alias = TOP_SENSOR_ALIAS if which == "top" else SIDE_SENSOR_ALIAS
    return Response(mjpeg_generator_raw(CSIM_HOST, CSIM_PORT, alias, fps=FPS, stepped=STEPPED_RAW),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

#====================== Inference and drawing ======================
@app.route("/snapshot.jpg")
def snapshot_image():
    frame = grabber.get_frame()
    if frame is None: abort(503, "no frame yet")
    if (not TORCH_OK) or (DETECTOR is None):
        jpg = encode_jpeg(frame, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    poly_color = _hex_to_rgb(request.args.get("poly_color", "#00dcdc"))
    poly_width = request.args.get("poly_width", default=3, type=int)

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

        vis = draw_dets(img_pil, boxes, scores, labels, cnames,
                        selected_index=SELECTED_DET_ID, masks=masks_np,
                        poly_color=poly_color, poly_width=int(poly_width))
        buf = io.BytesIO(); vis.save(buf, format="JPEG", quality=int(JPEG_QUALITY))
        jpg = buf.getvalue()
    except Exception as e:
        dbg = frame.copy()
        cv2.putText(dbg, f"inference error: {e}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        jpg = encode_jpeg(dbg, quality=JPEG_QUALITY)

    resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

#====================== Inference and drawing ======================
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
        w = max(0.0, x2-x1); h = max(0.0, y2-y1)
        cx = x1 + 0.5*w; cy = y1 + 0.5*h
        lab_name = cnames[int(l)] if (0 <= int(l) < len(cnames)) else f"id_{int(l)}"
        dets.append({"id": i, "score": float(s), "label": int(l), "label_name": lab_name,
                     "box": [x1,y1,x2,y2], "cx": cx, "cy": cy, "w": w, "h": h, "area": w*h})
    return jsonify({"detections": dets, "selected_id": SELECTED_DET_ID,
                    "model": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
                    "ts": LAST_DETS.get("ts"), "size": LAST_DETS.get("image_size")})

#====================== Select detection ======================
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
        if idx is None: return jsonify({"ok": False, "error": "no id provided"}), 400
        idx = int(idx)
        n = len(LAST_DETS["boxes"]) if LAST_DETS else 0
        if idx < 0 or idx >= n:
            return jsonify({"ok": False, "error": f"id {idx} out of range (0..{max(0,n-1)})"}), 400
        SELECTED_DET_ID = idx
        return jsonify({"ok": True, "selected_id": SELECTED_DET_ID})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

#====================== Selected crop image ======================
@app.route("/selected_crop.jpg")
def selected_crop():
    try:
        det_id = request.args.get("id", None, type=int)
        pad   = request.args.get("pad", default=0.08, type=float)
        outw  = request.args.get("size", default=None, type=int)
        crop_bgr, _ = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        jpg = encode_jpeg(crop_bgr, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((120,480,3), np.uint8)
        cv2.putText(err, f"crop err: {e}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        jpg = encode_jpeg(err, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

#====================== Mask image ======================
@app.route("/mask.jpg")
def mask_image():
    try:
        det_id = request.args.get("id", None, type=int)
        pad   = request.args.get("pad", default=0.08, type=float)
        outw  = request.args.get("size", default=None, type=int)
        _, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        if crop_mask.ndim == 3:
            crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
        jpg = encode_jpeg(vis, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp
    except Exception as e:
        err = np.zeros((160,680,3), np.uint8)
        cv2.putText(err, f"mask error: {e}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        jpg = encode_jpeg(err, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

#====================== Pattern image ======================
@app.route("/pattern.jpg")
def pattern_image():
    try:
        det_id   = request.args.get("id", None, type=int)
        pad      = request.args.get("pad",  default=0.08, type=float)
        outw     = request.args.get("size", default=None, type=int)
        style    = request.args.get("pattern", default="continuous", type=str).lower()
        spacing  = request.args.get("spacing", default=20, type=int)
        curvature_gain = request.args.get("curvature_gain", default=18.0, type=float)
        outside_scale  = request.args.get("outside_scale",  default=0.12, type=float)
        border_push_px = request.args.get("border_push_px", default=0.0,  type=float)
        bez_samples    = request.args.get("bez_samples",    default=16,   type=int)
        thread_color   = _hex_to_bgr(request.args.get("thread_color", "#1ed5ff"))
        thread_thick   = request.args.get("thread_thick", default=2, type=int)
        entry_mm       = request.args.get("entry_mm",  default=6.0, type=float)
        s_min          = request.args.get("s_min", default=8,  type=int)
        s_max          = request.args.get("s_max", default=60, type=int)
        px_per_mm_q    = request.args.get("px_per_mm", default=None, type=float)

        # convert color to a scalar so stitching can safely int(color)
        thread_color_scalar = _color_to_scalar(thread_color)

        # clamp 5–10 mm
        entry_mm = float(min(10.0, max(5.0, entry_mm)))

        crop_bgr, crop_mask = _get_selected_crop_and_mask_np(det_id, pad=pad, outw=outw)
        if crop_mask is None or crop_mask.size == 0:
            crop_mask = np.zeros(crop_bgr.shape[:2], np.uint8)
        elif crop_mask.ndim == 3:
            crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

        H, W = crop_bgr.shape[:2]
        ppm_fallback = max(0.0, (float(W)/120.0 + float(H)/175.0) * 0.5)
        # IMPORTANT: pass W, H as separate numbers (your mapper expects that)
        px_per_mm = (
            float(px_per_mm_q)
            if (px_per_mm_q not in (None, ""))
            else get_px_per_mm(PAD_MAPPER, W, H, ppm_fallback)
        )

        max_probe = int(max(crop_mask.shape) * 2.0)
        common_kwargs = dict(
            spacing_px=float(spacing),
            max_probe=int(max_probe),
            color_thread=thread_color_scalar,
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
                color=thread_color_scalar, thickness=int(thread_thick),
                curvature_gain=float(curvature_gain), outside_scale=float(outside_scale),
                spur_min_px=4, entry_mm=float(entry_mm), px_per_mm=float(px_per_mm) if px_per_mm else None,
            )
        elif style == "mold":
            over, _ = stitching.draw_stitching_mold_border(
                crop_bgr, crop_mask, grow_px=3, border_push_px=float(border_push_px), **common_kwargs
            )
        elif style == "bezier":
            fn = getattr(stitching, "draw_stitching_bezier", None)
            if fn is None:
                over, _ = stitching.draw_running_suture_auto(
                    crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
                )
            else:
                over, _ = fn(
                    crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0,
                    bezier_samples_per=int(bez_samples), **common_kwargs
                )
        else:
            over, _ = stitching.draw_running_suture_auto(
                crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
            )

        jpg = encode_jpeg(over, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        err = np.zeros((220,980,3), np.uint8)
        cv2.putText(err, f"pattern error: {e}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        jpg = encode_jpeg(err, quality=JPEG_QUALITY)
        resp = make_response(jpg); resp.headers["Content-Type"] = "image/jpeg"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

#====================== Robot movement ======================
@app.route("/move_robot", methods=["POST"])
def move_robot():
    """
    Build a suture polyline from the selected cut (in the crop),
    convert to MAT-frame meters (origin at mat center, +Y up), and publish on /suture_cuts.

    Notes:
      - crop pixels → add crop offset → FULL pixels
      - center around image center (≈ /mat origin)
      - flip Y: image v grows down, /mat +Y is up
      - px/mm fallback uses your mat size: 297.5 mm × 425.0 mm
    """
    print("[vision_web] /move_robot called", flush=True)

    # ---------- load ROS msg type (lazy) ----------
    try:
        from std_msgs.msg import String as _MsgString
    except Exception as e:
        return jsonify({"ok": False, "error": f"ROS2 std_msgs import failed: {e}"}), 500

    # ---------- request payload ----------
    payload = {}
    if request.is_json:
        try:
            payload = request.get_json(force=True) or {}
        except Exception:
            payload = {}

    # helpers
    def _f(x, default=None):
        try:
            if x in (None, ""): return default
            return float(x)
        except Exception:
            return default

    def _i(x, default=None):
        try:
            if x in (None, ""): return default
            return int(x)
        except Exception:
            return default

    det_id         = _i(payload.get("det_id"), _i(SELECTED_DET_ID, -1))
    style          = str(payload.get("pattern", "continuous")).lower()
    spacing_px     = _f(payload.get("spacing"), 20.0)
    entry_mm       = _f(payload.get("entry_mm"), 6.0)
    depth_mm       = _f(payload.get("depth_mm"), 3.0)
    pad            = _f(payload.get("pad"), 0.08)
    outw           = _i(payload.get("size"), 320)
    curvature_gain = _f(payload.get("curvature_gain"), 18.0)
    bez_samples    = _i(payload.get("bez_samples"), 16)
    outside_scale  = _f(payload.get("outside_scale"), 0.12)
    thread_thick   = _i(payload.get("thread_thick"), 2)
    px_per_mm_ui   = _f(payload.get("px_per_mm"), None)

    # clinical clamp 5–10 mm
    entry_mm = max(5.0, min(10.0, entry_mm))

    if det_id is None or det_id < 0:
        return jsonify({"ok": False, "error": "No detection selected"}), 400

    # ---------- crop + mask + meta (offsets) ----------
    try:
        crop_bgr, crop_mask, xx1, yy1, W_full, H_full = _get_selected_crop_mask_and_meta(det_id, pad=pad, outw=outw)
    except Exception as e:
        return jsonify({"ok": False, "error": f"crop/mask error: {e}"}), 400

    if crop_mask is None or crop_mask.size == 0:
        return jsonify({"ok": False, "error": "Empty mask/crop"}), 400
    if crop_mask.ndim == 3:
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

    Hc, Wc = crop_bgr.shape[:2]
    max_probe = int(max(crop_mask.shape) * 2.0)

    # ---------- build 2D path in image pixels (CROP coords) ----------
    common_kwargs = dict(
        spacing_px=float(spacing_px),
        max_probe=int(max_probe),
        color_thread=_color_to_scalar((30, 200, 255)),
        thickness=int(thread_thick),
        debug=False,
        curvature_gain=float(curvature_gain),
        outside_scale=float(outside_scale),
        spur_min_px=4,
        entry_mm=float(entry_mm),
        px_per_mm=None,  # drawing doesn't need true scale
    )

    try:
        if style == "perp":
            _, path2d = stitching.draw_stitching_pattern(
                crop_bgr, crop_mask, info={}, spacing=int(spacing_px),
                length_scale=2.0, color=_color_to_scalar((30, 200, 255)), thickness=int(thread_thick),
                curvature_gain=float(curvature_gain), outside_scale=float(outside_scale),
                spur_min_px=4, entry_mm=float(entry_mm), px_per_mm=None,
            )
        elif style == "mold":
            _, path2d = stitching.draw_stitching_mold_border(
                crop_bgr, crop_mask, grow_px=3, border_push_px=0.0, **common_kwargs
            )
        elif style == "bezier":
            draw_fn = getattr(stitching, "draw_stitching_bezier", None)
            if draw_fn is None:
                _, path2d = stitching.draw_running_suture_auto(
                    crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
                )
            else:
                _, path2d = draw_fn(
                    crop_bgr, crop_mask,
                    outside_px=3.0, rect_min_step=6.0,
                    bezier_samples_per=int(bez_samples),
                    **common_kwargs,
                )
        else:
            # default: continuous (running suture)
            _, path2d = stitching.draw_running_suture_auto(
                crop_bgr, crop_mask, outside_px=3.0, rect_min_step=6.0, **common_kwargs
            )
    except Exception as e:
        return jsonify({"ok": False, "error": f"path gen error: {e}"}), 500

    if not path2d:
        return jsonify({"ok": False, "error": "No path produced from the selected cut"}), 400

    # ---------- crop-px → full-px → (centered) → meters in /mat ----------
    # Prefer UI px/mm; else mapper using FULL image size; else physical-size fallback (297.5 x 425 mm)
    if px_per_mm_ui is not None and px_per_mm_ui > 0:
        px_per_mm = float(px_per_mm_ui)
    else:
        MAT_W_MM = 297.5  # along image width
        MAT_H_MM = 425.0  # along image height
        ppm_fallback = ((float(W_full) / MAT_W_MM) + (float(H_full) / MAT_H_MM)) * 0.5 if (W_full > 0 and H_full > 0) else 3.0
        try:
            from .image_to_map_mapper import get_px_per_mm as _gppm
            px_per_mm = _gppm(PAD_MAPPER, W_full, H_full, ppm_fallback)
        except Exception:
            px_per_mm = ppm_fallback
    if not px_per_mm or px_per_mm <= 0:
        MAT_W_MM = 297.5
        MAT_H_MM = 425.0
        px_per_mm = ((float(W_full) / MAT_W_MM) + (float(H_full) / MAT_H_MM)) * 0.5 if (W_full > 0 and H_full > 0) else 3.0

    # full-image center in pixels (≈ mat center; camera is top-down over mat)
    cx = 0.5 * float(W_full)
    cy = 0.5 * float(H_full)

    def croppx_to_mat_m(u_crop: float, v_crop: float) -> Tuple[float, float]:
        # 1) crop → FULL image pixels
        u_full = xx1 + float(u_crop)
        v_full = yy1 + float(v_crop)
        # 2) recenter at image center (= mat center)
        u_c = u_full - cx
        v_c = v_full - cy
        # 3) flip Y (image down vs. mat +Y up)
        x_mm =  u_c / px_per_mm
        y_mm = -v_c / px_per_mm
        return (x_mm / 1000.0, y_mm / 1000.0)

    poly_m = [croppx_to_mat_m(u, v) for (u, v) in path2d]

    spacing_m = float(spacing_px) / px_per_mm / 1000.0
    entry_m   = float(entry_mm) / 1000.0
    depth_m   = float(depth_mm) / 1000.0

    cuts_msg = {
        "frame_id": "mat",
        "cuts": [ {"polyline": [[x, y] for (x, y) in poly_m]} ],
        "params": {
            "spacing":  max(1e-4, spacing_m),
            "entry_mm": entry_m,   # NOTE: consumer expects meters though key says "mm"
            "depth":    depth_m
        }
    }

    # ---------- publish ----------
    global cuts_pub
    if cuts_pub is None:
        try:
            init_ros2_publisher()
        except Exception:
            pass
    if cuts_pub is None:
        return jsonify({"ok": False, "error": "ROS2 publisher not initialized"}), 500

    msg = _MsgString()
    msg.data = json.dumps(cuts_msg)
    cuts_pub.publish(msg)
    print(f"[vision_web] published /suture_cuts with {len(poly_m)} points (px/mm={px_per_mm:.3f})", flush=True)

    # --------- Publish normalized coordinates on /vision_targets ---------
    try:
        # Ensure the vision targets publisher exists.
        global targets_pub
        if targets_pub is None:
            init_ros2_publisher()
        # Only proceed if publisher is available and we have a path.
        if targets_pub is not None and path2d:
            norm_targets = []
            # Normalize path coordinates to range [0,1] in both axes.  Use full image
            # coordinates (crop offset + point) relative to the full sensor view so
            # that (0,0) corresponds to the top-left of the full 1m×1m sensor and
            # (1,1) to the bottom-right.  This ensures a consistent mapping
            # independent of the bounding box size.  See lap_ik mapping for how
            # these unit coordinates are converted back into world-frame values.
            for (u_crop, v_crop) in path2d:
                try:
                    # Convert crop-relative pixel positions into absolute full-image
                    # coordinates by adding the crop offset (xx1, yy1), then
                    # normalize by the full image dimensions (W_full, H_full).
                    u_full = xx1 + float(u_crop)
                    v_full = yy1 + float(v_crop)
                    x_norm = u_full / float(W_full) if W_full > 0 else 0.0
                    y_norm = v_full / float(H_full) if H_full > 0 else 0.0
                except Exception:
                    x_norm, y_norm = 0.0, 0.0
                # Clamp to [0,1]
                if x_norm < 0.0: x_norm = 0.0
                elif x_norm > 1.0: x_norm = 1.0
                if y_norm < 0.0: y_norm = 0.0
                elif y_norm > 1.0: y_norm = 1.0
                norm_targets.append({"pos": [x_norm, y_norm]})
            # Print normalized path for debugging/verification.
            try:
                print("[vision_web] normalized path (x_norm, y_norm): " +
                      str([(float(t['pos'][0]), float(t['pos'][1])) for t in norm_targets]), flush=True)
            except Exception:
                pass
            # Publish the normalized targets on /vision_targets in the 'unit' frame.
            vt_payload = {"frame": "unit", "targets": norm_targets}
            vt_msg = _MsgString()
            vt_msg.data = json.dumps(vt_payload)
            targets_pub.publish(vt_msg)
            print(f"[vision_web] published /vision_targets with {len(norm_targets)} normalized points", flush=True)
    except Exception as e:
        print(f"[vision_web][WARN] failed to publish /vision_targets: {e}", flush=True)

    return jsonify({
        "ok": True,
        "published": True,
        "topic": "/suture_cuts",
        "n_poly_pts": len(poly_m),
        "spacing_m": spacing_m,
        "entry_m": entry_m,
        "depth_m": depth_m,
        "px_per_mm": px_per_mm
    })
# ====================== Health ======================
@app.route("/health")
def health():
    ok = grabber.get_frame() is not None
    return ({
        "frame": ok,
        "torch": TORCH_OK,
        "model_loaded": (DETECTOR is not None),
        "model_name": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
    }, 200 if ok else 503)

# ====================== Main ======================

def main():
    print(f"[vision_web] templates dir: {TEMPLATES_DIR}", flush=True)
    print(f"[vision_web] models root  : {MODELS_ROOT}", flush=True)

    # Start grabber
    grabber.start()
    t0 = time.time()
    while grabber.get_frame() is None and (time.time() - t0) < 5.0:
        time.sleep(0.05)

    # Load optional calibration
    global PAD_MAPPER
    PAD_MAPPER = load_pad_mapper(CALIB_PATH)  # your module handles "", missing files, etc.

    # ROS2 pub
    init_ros2_publisher()

    # Detector
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