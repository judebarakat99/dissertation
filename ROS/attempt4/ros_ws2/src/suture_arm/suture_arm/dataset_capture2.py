#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset capture from CoppeliaSim with domain randomization:
- Moves UR3 through small jitters (occlusions/variety)
- Randomizes lights (diffuse/specular, orientation, occasional on/off)
- Randomizes pad color (shapes under the 'mat' dummy)
- Captures raw JPEGs + overlay thumbnails + JSONL metadata
- (Optional) small camera jitter per frame

Env / defaults:
  CSIM_HOST=127.0.0.1
  CSIM_PORT=23000
  VISION_SENSOR=/visionSensor
  UR_BASE=/UR3
  MAT_ALIAS=/mat
  OUT_DIR=./dataset_out
  N=200                       # frames to capture
  JPEG_QUALITY=92
  STEPPED=1                   # setStepping + sim.step()
  CAM_JITTER_DEG=2.0          # camera pose jitter (±deg)
  LIGHT_JITTER_DEG=15.0       # light orientation jitter (±deg)
  LIGHT_ON_P=0.85             # prob. a light is ON this frame
  PAD_COLOR_VARIANCE=0.25     # how much to vary pad color (0..1)
  SLEEP=0.02                  # small sleep per step (s)
"""

import os, time, json, math, random, traceback
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import cv2

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy

# -------------------- Config (env) --------------------
CSIM_HOST        = os.getenv("CSIM_HOST", "127.0.0.1")
CSIM_PORT        = int(os.getenv("CSIM_PORT", "23000"))
SENSOR_ALIAS     = os.getenv("VISION_SENSOR", "/visionSensor")
UR_BASE_ALIAS    = os.getenv("UR_BASE", "/UR3")
MAT_ALIAS        = os.getenv("MAT_ALIAS", "/mat")
OUT_DIR          = os.getenv("OUT_DIR", "./dataset_out")
N_FRAMES         = int(os.getenv("N", "200"))
JPEG_QUALITY     = int(os.getenv("JPEG_QUALITY", "92"))
STEPPED          = int(os.getenv("STEPPED", "1"))
CAM_JITTER_DEG   = float(os.getenv("CAM_JITTER_DEG", "2.0"))
LIGHT_JITTER_DEG = float(os.getenv("LIGHT_JITTER_DEG", "15.0"))
LIGHT_ON_P       = float(os.getenv("LIGHT_ON_P", "0.85"))
PAD_COLOR_VAR    = float(os.getenv("PAD_COLOR_VARIANCE", "0.25"))
SLEEP            = float(os.getenv("SLEEP", "0.02"))

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Small safe joint jitters (radians) to create occlusions without self-collision
JITTER = {
    "shoulder_pan_joint":   (-0.45, +0.45),
    "shoulder_lift_joint":  (-0.30, +0.30),
    "elbow_joint":          (-0.35, +0.35),
    "wrist_1_joint":        (-0.30, +0.30),
    "wrist_2_joint":        (-0.60, +0.60),
    "wrist_3_joint":        (-0.60, +0.60),
}

# Base “silicone pink” color (ambient+diffuse)
PAD_BASE_COLOR = np.array([0.98, 0.78, 0.82], dtype=float)

# -------------------- Helpers --------------------
def log(msg: str): print(f"[dataset_capture2] {msg}", flush=True)
def log_err(msg: str): print(f"[dataset_capture2][ERR] {msg}", flush=True)

def _writable(bgr: np.ndarray) -> np.ndarray:
    if not (bgr.flags['C_CONTIGUOUS'] and bgr.flags['WRITEABLE']):
        bgr = np.ascontiguousarray(bgr.copy())
    return bgr

# -------------------- Coppelia access --------------------
def _resolve(sim, alias: str) -> Optional[int]:
    for cand in (alias, alias.lstrip('/'), alias + '#0'):
        try: return sim.getObject(cand)
        except Exception: pass
    return None

def _get_sensor_frame(sim, sensor) -> np.ndarray:
    # explicit handling if needed
    try:
        if bool(sim.getObjectInt32Param(sensor, sim.visionintparam_explicit_handling)):
            sim.handleVisionSensor(sensor)
    except Exception:
        pass

    if STEPPED and hasattr(sim, "setStepping"):
        try: sim.step()
        except Exception: pass

    # Robust reads across versions:
    try:
        img, res = sim.getVisionSensorImg(sensor)  # (buf, [w,h])
        w, h = int(res[0]), int(res[1])
    except Exception:
        try:
            img, w, h = sim.getVisionSensorCharImage(sensor)
        except Exception as e1:
            try:
                out = sim.getVisionSensorImage(sensor)
                if isinstance(out, (list, tuple)) and len(out) == 3 and isinstance(out[1], (int, float)):
                    img, w, h = out; w, h = int(w), int(h)
                else:
                    img, res = out; w, h = int(res[0]), int(res[1])
                if isinstance(img, (list, tuple, np.ndarray)) and not isinstance(img, (bytes, bytearray)):
                    arr = np.array(img, dtype=np.float32)
                    img = (arr * 255).clip(0, 255).astype(np.uint8).tobytes()
            except Exception as e2:
                raise RuntimeError(f"getVisionSensor* failed: {e1} / {e2}")

    buf = np.frombuffer(img, dtype=np.uint8) if isinstance(img,(bytes,bytearray)) else np.array(img, dtype=np.uint8)
    n = buf.size
    if n == w*h:
        frame = cv2.cvtColor(np.flip(buf.reshape(h,w),0), cv2.COLOR_GRAY2BGR)
    elif n == w*h*3:
        frame = np.flip(buf.reshape(h,w,3),0)[:, :, ::-1]
    elif n == w*h*4:
        frame = np.flip(buf.reshape(h,w,4),0)[:, :, :3][:, :, ::-1]
    else:
        if (w*h) and n % (w*h) == 0:
            c = n // (w*h)
            frame = np.flip(buf.reshape(h,w,c),0)[:, :, :3][:, :, ::-1]
        else:
            raise RuntimeError(f"unexpected buffer size n={n} vs {w}x{h}")

    return _writable(frame)

def _get_T_A_from_B(sim, objA, objB) -> np.ndarray:
    """4x4 transform from B to A (T_A_B)."""
    m12 = sim.getObjectMatrix(objA, objB)
    M = np.eye(4); M[:3,:4] = np.array(m12, dtype=float).reshape(3,4)
    return np.linalg.inv(M)

# -------------------- Domain randomization --------------------
def _list_lights(sim) -> List[int]:
    LIGHT = getattr(sim, 'object_light_type', 1)
    try:
        return sim.getObjectsInTree(-1, LIGHT, 0)  # all lights in scene
    except Exception:
        return []

def _random_unit() -> float:
    return random.uniform(0.0, 1.0)

def _rand_color(min_v=0.15, max_v=1.0) -> List[float]:
    return [float(random.uniform(min_v, max_v)) for _ in range(3)]

def _rand_near_color(base_rgb: np.ndarray, var: float) -> List[float]:
    # add small Gaussian noise and clamp to [0,1]
    var = max(0.0, min(1.0, var))
    noisy = base_rgb + np.random.normal(0.0, var*0.25, size=3)
    return np.clip(noisy, 0.0, 1.0).astype(float).tolist()

def _randomize_lights(sim, lights: List[int]):
    """Randomize each light's on/off, color, and orientation slightly."""
    for lh in lights:
        try:
            # state: keep shadow/etc. flags, but toggle on/off by probability
            try:
                st, old_diff, old_spec = sim.getLightParameters(lh)
            except Exception:
                st, old_diff, old_spec = 1, [1,1,1], [0.3,0.3,0.3]

            on = 1 if (_random_unit() < LIGHT_ON_P) else 0
            state = int(bool(on))

            diff = _rand_color(0.4, 1.0)        # brighter diffuse
            spec = _rand_color(0.0, 0.6)        # softer specular
            sim.setLightParameters(lh, state, diff, spec)

            # small direction jitter (± LIGHT_JITTER_DEG)
            if LIGHT_JITTER_DEG > 0.0:
                rpy = sim.getObjectOrientation(lh, -1)
                rpy = [
                    rpy[0] + math.radians(random.uniform(-LIGHT_JITTER_DEG, LIGHT_JITTER_DEG)),
                    rpy[1] + math.radians(random.uniform(-LIGHT_JITTER_DEG, LIGHT_JITTER_DEG)),
                    rpy[2] + math.radians(random.uniform(-LIGHT_JITTER_DEG, LIGHT_JITTER_DEG)),
                ]
                sim.setObjectOrientation(lh, -1, rpy)
        except Exception:
            # ignore per-light failures
            pass

def _shapes_under(sim, base_handle: int) -> List[int]:
    SHAPE = getattr(sim, 'object_shape_type', 4)
    shapes = []
    try:
        sim.getShapeMesh(base_handle); shapes=[base_handle]
    except Exception:
        pass
    if not shapes:
        try:
            shapes = sim.getObjectsInTree(base_handle, SHAPE, 1)
        except Exception:
            shapes = []
    return shapes

def _randomize_pad_color(sim, mat_handle: Optional[int]):
    """Randomize colors of any shapes under 'mat'."""
    if mat_handle is None: return
    shapes = _shapes_under(sim, mat_handle)
    if not shapes: return
    for sh in shapes:
        try:
            # ambient+diffuse component
            color = _rand_near_color(PAD_BASE_COLOR, PAD_COLOR_VAR)
            sim.setShapeColor(sh, None, getattr(sim, 'colorcomponent_ambient_diffuse', 0), color)
            # a bit of emission/specular variance
            spec = _rand_color(0.0, 0.5)
            sim.setShapeColor(sh, None, getattr(sim, 'colorcomponent_specular', 1), spec)
            emi = _rand_color(0.0, 0.1)
            sim.setShapeColor(sh, None, getattr(sim, 'colorcomponent_emission', 2), emi)
        except Exception:
            pass

def _cam_jitter(sim, sensor):
    if CAM_JITTER_DEG <= 0.0: return
    try:
        ori = sim.getObjectOrientation(sensor, -1)
        r = [ori[0] + math.radians(random.uniform(-CAM_JITTER_DEG, CAM_JITTER_DEG)),
             ori[1] + math.radians(random.uniform(-CAM_JITTER_DEG, CAM_JITTER_DEG)),
             ori[2] + math.radians(random.uniform(-CAM_JITTER_DEG, CAM_JITTER_DEG))]
        sim.setObjectOrientation(sensor, -1, r)
    except Exception:
        pass

# -------------------- UR3 movement --------------------
def _get_ur_joints(sim, base_alias: str, names: List[str]) -> List[int]:
    base = _resolve(sim, base_alias)
    if base is None:
        raise RuntimeError(f"UR base not found: {base_alias}")
    handles=[]
    missing=[]
    for n in names:
        h=None
        for cand in (f"{base_alias}/{n}", n):
            try:
                h = sim.getObject(cand); break
            except Exception: pass
        if h is None: missing.append(n)
        else: handles.append(h)
    if missing:
        raise RuntimeError(f"Missing UR joints under {base_alias}: {missing}")
    return handles

def _get_q(sim, joints: List[int]) -> np.ndarray:
    return np.array([sim.getJointPosition(h) for h in joints], float)

def _goto(sim, joints: List[int], q_target: np.ndarray, steps: int = 50, sleep: float = SLEEP):
    q0 = _get_q(sim, joints)
    for s in range(1, steps+1):
        q = q0 + (s/steps)*(q_target - q0)
        for h, qi in zip(joints, q):
            sim.setJointTargetPosition(h, float(qi))
        if STEPPED and hasattr(sim,"setStepping"):
            sim.step()
        time.sleep(sleep)

def _jitter_pose(q0: np.ndarray) -> np.ndarray:
    q = q0.copy()
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = JITTER.get(name, (-0.1,0.1))
        q[i] = q0[i] + random.uniform(lo, hi)
    return q

# -------------------- Homography (optional metadata) --------------------
def _compute_pad_homography(sim, sensor, mat) -> Optional[np.ndarray]:
    try:
        # find biggest bbox under mat
        shapes = _shapes_under(sim, mat)
        if not shapes: return None
        def to4x4(m12): T=np.eye(4); T[:3,:4]=np.array(m12,float).reshape(3,4); return T
        def bbox_in_mat(h):
            def grab(prefix):
                try:
                    mnx=sim.getObjectFloatParam(h, getattr(sim, prefix+'_min_x'))
                    mny=sim.getObjectFloatParam(h, getattr(sim, prefix+'_min_y'))
                    mnz=sim.getObjectFloatParam(h, getattr(sim, prefix+'_min_z'))
                    mxx=sim.getObjectFloatParam(h, getattr(sim, prefix+'_max_x'))
                    mxy=sim.getObjectFloatParam(h, getattr(sim, prefix+'_max_y'))
                    mxz=sim.getObjectFloatParam(h, getattr(sim, prefix+'_max_z'))
                    return [mnx,mny,mnz,mxx,mxy,mxz]
                except Exception: return None
            bb = grab('objfloatparam_objbbox') or grab('objfloatparam_modelbbox')
            if bb is None: return None
            xmin,ymin,zmin,xmax,ymax,zmax = map(float, bb)
            corners = np.array([
                [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin]
            ], float)
            T = to4x4(sim.getObjectMatrix(h, mat))
            return (T @ np.c_[corners, np.ones((4,1))].T).T[:, :3]
        best=None; bestA=-1
        for h in shapes:
            V = bbox_in_mat(h)
            if V is None or V.size==0: continue
            xs, ys = V[:,0], V[:,1]
            A = (xs.max()-xs.min())*(ys.max()-ys.min())
            if A>bestA:
                bestA=A; best=(xs.min(),xs.max(),ys.min(),ys.max())
        if best is None: return None
        xmin,xmax,ymin,ymax = best
        C = np.array([[xmin,ymin,0],[xmax,ymin,0],[xmax,ymax,0],[xmin,ymax,0]], float)

        # mat->cam
        T_cam_mat = _get_T_A_from_B(sim, sensor, mat)
        Pc = (T_cam_mat @ np.c_[C, np.ones((4,1))].T).T[:, :3]

        # image size
        frame = _get_sensor_frame(sim, sensor)
        H, W = frame.shape[:2]
        try:
            vfov = float(sim.getObjectFloatParam(sensor, sim.visionfloatparam_perspective_angle))
        except Exception:
            vfov = math.radians(60.0)
        cx, cy = 0.5*W, 0.5*H
        fy = (H*0.5) / math.tan(0.5*vfov)
        fx = fy * (W/H)
        uv=[]
        for x,y,z in Pc:
            if z<=1e-6: return None
            uv.append([fx*(x/z)+cx, -fy*(y/z)+cy])
        P = np.array(uv, dtype=np.float32)
        src = C[:, :2].astype(np.float32)
        return cv2.getPerspectiveTransform(src, P)
    except Exception:
        return None

# -------------------- Main --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, "images"); os.makedirs(img_dir, exist_ok=True)
    ovl_dir = os.path.join(OUT_DIR, "overlay"); os.makedirs(ovl_dir, exist_ok=True)
    meta_path = os.path.join(OUT_DIR, "meta.jsonl")

    client = RemoteAPIClient(CSIM_HOST, CSIM_PORT)
    sim = client.require('sim')

    # ensure running
    try:
        st = sim.getSimulationState()
        if st in (sim.simulation_stopped, sim.simulation_paused):
            sim.startSimulation()
    except Exception: pass

    # stepped mode if desired
    if STEPPED and hasattr(sim, "setStepping"):
        try: sim.setStepping(True)
        except Exception: pass

    # handles
    sensor = _resolve(sim, SENSOR_ALIAS)
    if sensor is None:
        raise RuntimeError(f"vision sensor not found: {SENSOR_ALIAS}")
    mat = _resolve(sim, MAT_ALIAS)  # may be None
    joints = _get_ur_joints(sim, UR_BASE_ALIAS, JOINT_NAMES)
    q_home = _get_q(sim, joints)

    # discover lights once
    lights = _list_lights(sim)
    log(f"Found {len(lights)} light(s)")

    # precompute homography (best-effort)
    Hmat = _compute_pad_homography(sim, sensor, mat) if mat is not None else None
    if Hmat is None:
        log("WARN: could not compute homography; labels will lack H_mat_to_image")

    # capture loop
    with open(meta_path, "a") as mf:
        for i in range(N_FRAMES):
            try:
                # --- domain randomization ---
                _randomize_lights(sim, lights)
                _randomize_pad_color(sim, mat)
                _cam_jitter(sim, sensor)

                # --- move UR for occlusions ---
                q_target = _jitter_pose(q_home)
                _goto(sim, joints, q_target, steps=40, sleep=SLEEP)

                # --- capture frame ---
                frame = _get_sensor_frame(sim, sensor)
                H, W = frame.shape[:2]

                # save raw
                img_name = f"frame_{i:06d}.jpg"
                img_path = os.path.join(img_dir, img_name)
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
                if not ok: raise RuntimeError("cv2.imencode failed")
                with open(img_path, "wb") as f: f.write(enc.tobytes())

                # overlay thumb
                ovl = frame.copy()
                cv2.line(ovl, (W//2-20,H//2), (W//2+20,H//2), (200,200,200), 1, cv2.LINE_AA)
                cv2.line(ovl, (W//2,H//2-20), (W//2,H//2+20), (200,200,200), 1, cv2.LINE_AA)
                cv2.putText(ovl, f"{i+1}/{N_FRAMES}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,240,40), 2, cv2.LINE_AA)
                ovl_path = os.path.join(ovl_dir, f"frame_{i:06d}_overlay.jpg")
                ok, enc2 = cv2.imencode(".jpg", ovl, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if ok:
                    with open(ovl_path, "wb") as f: f.write(enc2.tobytes())

                # metadata
                q_now = _get_q(sim, joints).tolist()
                T_cam_world = _get_T_A_from_B(sim, sensor, -1).tolist()
                T_mat_world = _get_T_A_from_B(sim, mat, -1).tolist() if mat is not None else None
                rec = {
                    "index": i,
                    "timestamp": time.time(),
                    "image": os.path.relpath(img_path, OUT_DIR),
                    "overlay": os.path.relpath(ovl_path, OUT_DIR),
                    "sensor_alias": SENSOR_ALIAS,
                    "ur_base_alias": UR_BASE_ALIAS,
                    "mat_alias": MAT_ALIAS if mat is not None else None,
                    "image_size": [int(W), int(H)],
                    "joint_names": JOINT_NAMES,
                    "q": q_now,
                    "T_cam_world": T_cam_world,
                    "T_mat_world": T_mat_world,
                    "H_mat_to_image": (np.array(Hmat).tolist() if Hmat is not None else None),
                }
                mf.write(json.dumps(rec) + "\n")
                if (i+1) % 10 == 0:
                    log(f"Captured {i+1}/{N_FRAMES}")

            except Exception as e:
                log_err(f"frame {i} failed: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    log(f"Done. Images → {img_dir}, overlays → {ovl_dir}, meta → {meta_path}")
    # return to home (best effort)
    try: _goto(sim, joints, q_home, steps=50, sleep=SLEEP)
    except Exception: pass

if __name__ == "__main__":
    main()
