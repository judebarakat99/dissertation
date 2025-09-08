# Minimal model wrapper for vision_web.py
# Must define:
#   load(model_path) -> handle
#   predict(handle, frame_bgr) -> {"params": {...}, "cuts": [{"polyline": [[x,y], ...]}]}
#
# - Returns polylines in the mat frame (meters, z=0).
# - Uses calib.json (optional) to convert pixels->meters.

import os
import json
import numpy as np
import cv2

try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False


# ---------- Calibration helpers ----------

def _find_calib_path(model_path: str):
    # 1) same directory as model
    d = os.path.dirname(os.path.abspath(model_path))
    p = os.path.join(d, "calib.json")
    if os.path.isfile(p):
        return p
    # 2) ML_detection directory
    dd = os.path.dirname(d)
    p2 = os.path.join(dd, "calib.json")
    if os.path.isfile(p2):
        return p2
    # 3) this file's directory
    here = os.path.dirname(__file__)
    p3 = os.path.join(here, "calib.json")
    if os.path.isfile(p3):
        return p3
    return None


def _load_calib(model_path: str):
    p = _find_calib_path(model_path)
    if p:
        try:
            with open(p, "r") as f:
                c = json.load(f)
            return c
        except Exception:
            pass
    # defaults if no calib present
    return {
        "scale_m_per_px": 0.0005,   # 0.5 mm per pixel default
        "origin_px": None,          # None -> center of image
        "y_down": True,
        "spacing": 0.006,
        "bite": 0.005
    }


def _pixels_to_mat(poly_px: np.ndarray, calib: dict, img_shape_hw):
    """
    Convert Nx2 pixel coords -> Nx2 meters in 'mat' frame, z=0.
    Supports either homography H (3x3) or scale/origin mapping.
    """
    H, W = img_shape_hw
    if poly_px is None or len(poly_px) == 0:
        return np.empty((0, 2), dtype=float)

    if "H" in calib:
        Hmat = np.array(calib["H"], dtype=float).reshape(3, 3)
        uv1 = np.c_[poly_px[:, 0], poly_px[:, 1], np.ones((len(poly_px), 1))]
        xyw = (Hmat @ uv1.T).T
        w = np.where(np.abs(xyw[:, 2]) < 1e-9, 1.0, xyw[:, 2])
        x = xyw[:, 0] / w
        y = xyw[:, 1] / w
        return np.c_[x, y]

    # scale/origin fallback
    scale = float(calib.get("scale_m_per_px", 0.0005))
    origin = calib.get("origin_px", None)
    if origin is None:
        origin = [W / 2.0, H / 2.0]
    ox, oy = float(origin[0]), float(origin[1])
    y_down = bool(calib.get("y_down", True))

    u = poly_px[:, 0]
    v = poly_px[:, 1]
    x = (u - ox) * scale
    y_pix = (v - oy)
    y = (-y_pix if y_down else y_pix) * scale
    return np.c_[x, y]


# ---------- Simple classical fallback (if model absent or fails) ----------

def _infer_fallback(frame_bgr: np.ndarray):
    """
    Returns a simple 2-point polyline in pixels by detecting the longest line.
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0,
                            threshold= max(60, min(H, W)//8),
                            minLineLength= max(40, min(H, W)//4),
                            maxLineGap= max(10, min(H, W)//20))
    if lines is None:
        return None
    # pick the longest
    best = None
    best_len = -1.0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        L = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if L > best_len:
            best_len = L
            best = (x1, y1, x2, y2)
    if best is None:
        return None
    x1, y1, x2, y2 = best
    return np.array([[x1, y1], [x2, y2]], dtype=float)


# ---------- Torch inference (best-effort generic) ----------

def _infer_with_torch(net, frame_bgr: np.ndarray):
    """
    Try to run TorchScript model. We assume the model returns either:
      - a mask-like tensor (B,1,H,W) or (B,H,W) or (H,W),
      - OR any tensor we can squeeze to (H,W).
    We then extract a main line from the mask with Hough.
    Customize this to your model’s API if needed.
    """
    if net is None or not _TORCH_OK:
        return None
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tens = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1,3,H,W)
        net.eval()
        with torch.no_grad():
            out = net(tens)

        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits

        if torch.is_tensor(out):
            arr = out.detach().cpu().float()
            # squeeze to (H,W)
            while arr.dim() > 2:
                arr = arr.squeeze(0)
            if arr.dim() != 2:
                return None
            arr = arr.numpy()
        else:
            arr = np.array(out)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                return None

        # Normalize and threshold
        mm = arr.max() - arr.min()
        if mm < 1e-9:
            return None
        norm = (arr - arr.min()) / mm
        mask = (norm > 0.5).astype(np.uint8) * 255

        # extract main line
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180.0,
                                threshold= max(60, mask.shape[0]//8),
                                minLineLength= max(40, mask.shape[0]//4),
                                maxLineGap= max(10, mask.shape[0]//20))
        if lines is None:
            return None
        best = None
        best_len = -1.0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            L = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if L > best_len:
                best_len = L
                best = (x1, y1, x2, y2)
        if best is None:
            return None
        x1, y1, x2, y2 = best
        return np.array([[x1, y1], [x2, y2]], dtype=float)

    except Exception:
        return None


# ---------- Public API ----------

def load(model_path: str):
    """
    Load the model (TorchScript preferred). Returns a handle dict with:
      {"net": model_or_None, "calib": calib_dict}
    """
    calib = _load_calib(model_path)
    net = None
    if _TORCH_OK:
        try:
            net = torch.jit.load(model_path, map_location="cpu")
            net.eval()
        except Exception:
            # Could add state_dict path here if you instantiate your own nn.Module
            net = None
    return {"net": net, "calib": calib}


def predict(handle, frame_bgr: np.ndarray):
    """
    Run detection; return dict in mat frame:
      {"params": {"spacing":..., "bite":...},
       "cuts": [{"polyline": [[x,y], ...]}, ...]}
    """
    calib = handle.get("calib", {})
    net = handle.get("net", None)

    # 1) Try your model
    poly_px = _infer_with_torch(net, frame_bgr)

    # 2) Fallback to classical if model not usable
    if poly_px is None:
        poly_px = _infer_fallback(frame_bgr)

    cuts = []
    if poly_px is not None:
        poly_mat = _pixels_to_mat(poly_px, calib, frame_bgr.shape[:2])
        cuts.append({"polyline": poly_mat.tolist()})

    params = {
        "spacing": float(calib.get("spacing", 0.006)),
        "bite": float(calib.get("bite", 0.005))
    }
    return {"params": params, "cuts": cuts}
