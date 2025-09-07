#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as MsgString

# Kinematics
import tf_transformations as tft
from ikpy.chain import Chain

# CoppeliaSim ZMQ Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback

# ----------------- Defaults / Tunables -----------------
CSIM_HOST_DEFAULT = "127.0.0.1"
CSIM_PORT_DEFAULT = 23000

ROBOT_BASE_PATH_DEFAULT = "/UR3"     # UR3 root in the scene
SUTURE_FRAME_PATH_DEFAULT = "/mat"   # dummy 'mat' (parent of suture_pad)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Mat physical size (meters). Your pad is ~297.5 mm x 425 mm (centered origin).
MAT_HALF_X = 0.2975 * 0.5  # along mat X (m)
MAT_HALF_Y = 0.4250 * 0.5  # along mat Y (m)

# Approach / motion shaping (meters / radians)
APPROACH_Z_DEFAULT = 0.012          # 12 mm above surface (in mat-Z)
DEPTH_DEFAULT      = 0.003          # 3 mm into the mat (negative mat-Z)
LINEAR_STEP_RAD    = 0.01           # max joint change per streamed step
SETTLE_TIME_SEC    = 0.004
POSE_TILT_RAD      = 0.20           # small pitch to avoid straight wrist (ignored when position-only)

# Preferred seed posture (UR3 elbow-up-ish)
NOMINAL_Q = np.array([0.0, -1.35, 1.90, 0.0, 1.30, 0.0])

# Soft joint limits (adjust if your scene needs more freedom)
SOFT_LIMITS = np.array([
    [-math.pi,  math.pi],  # J1
    [-2.60,     -0.10],    # J2
    [ 0.00,      3.00],    # J3
    [-2.50,      2.50],    # J4
    [ 0.25,      2.80],    # J5 (avoid wrist straight)
    [-math.pi,  math.pi],  # J6
], dtype=float)
TWOPI = 2.0 * math.pi
# -------------------------------------------------------


@dataclass
class PoseRPY:
    xyz: np.ndarray                 # (3,) in /mat
    rpy: Tuple[float, float, float] # roll, pitch, yaw (mat frame); ignored for position-only IK


# ----------------- Small helpers -----------------
def rpy_to_T(xyz: np.ndarray, rpy: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = tft.euler_matrix(*rpy)[:3, :3]
    T[:3,  3] = xyz
    return T

def wrap_to_nearest(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    out = q_target.copy()
    for i in range(len(out)):
        d = out[i] - q_ref[i]
        d = (d + math.pi) % TWOPI - math.pi
        out[i] = q_ref[i] + d
    return out

def clamp_soft(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(6):
        lo, hi = SOFT_LIMITS[i]
        q[i] = min(max(q[i], lo), hi)
    return q

def blend_seed(q_seed: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    return alpha * q_seed + (1.0 - alpha) * NOMINAL_Q

def clamp_xy_to_mat(xy: np.ndarray) -> np.ndarray:
    """Clamp a single XY point to the centered mat rectangle with 1 mm margin."""
    return np.array([
        float(np.clip(xy[0], -MAT_HALF_X + 1e-3, MAT_HALF_X - 1e-3)),
        float(np.clip(xy[1], -MAT_HALF_Y + 1e-3, MAT_HALF_Y - 1e-3))
    ], dtype=float)

def sample_polyline(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Resample Nx2 (or Nx3) polyline at ~equal arc distance 'spacing' (meters)."""
    if poly.shape[1] == 2:
        P = np.c_[poly, np.zeros(len(poly))]
    else:
        P = poly.copy()
    diffs = np.diff(P[:, :2], axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    dist = np.cumsum(np.r_[0.0, seglen])
    L = dist[-1]
    if L < 1e-9:
        return P[:1]
    s = np.arange(0.0, L, max(spacing, 1e-4))
    xs = np.interp(s, dist, P[:, 0])
    ys = np.interp(s, dist, P[:, 1])
    zs = np.interp(s, dist, P[:, 2])
    out = np.vstack([xs, ys, zs]).T
    if (L - s[-1]) > 1e-6:  # include last
        out = np.vstack([out, P[-1]])
    return out


# ----------------- IK wrapper for UR -----------------
class IKUR:
    def __init__(self, ur_type: str = "ur3"):
        # Expand URDF from xacro
        from ament_index_python.packages import get_package_share_directory
        ur_desc = get_package_share_directory("ur_description")
        xacro_path = os.path.join(ur_desc, "urdf", "ur.urdf.xacro")
        ur_type = ur_type.lower()
        valid = {"ur3","ur3e","ur5","ur5e","ur10","ur10e","ur16e","ur20"}
        if ur_type not in valid:
            raise ValueError(f"Unsupported UR type '{ur_type}'. Choose one of: {sorted(valid)}")
        cfg_dir = os.path.join(ur_desc, "config", ur_type)

        cmd = [
            "xacro", xacro_path,
            "name:=ur",
            f"ur_type:={ur_type}",
            f"kinematics_params:={os.path.join(cfg_dir, 'default_kinematics.yaml')}",
            f"joint_limit_params:={os.path.join(cfg_dir, 'joint_limits.yaml')}",
            f"physical_params:={os.path.join(cfg_dir, 'physical_parameters.yaml')}",
            f"visual_params:={os.path.join(cfg_dir, 'visual_parameters.yaml')}",
        ]
        xml = subprocess.check_output(cmd).decode("utf-8")
        # Patch continuous->revolute & add wide limits (ikpy requirement)
        xml = xml.replace('type="continuous"', 'type="revolute"')
        if "<limit" not in xml:
            xml = xml.replace(
                "</joint>",
                '<limit lower="-6.283185307179586" upper="6.283185307179586" velocity="3.0" effort="50.0"/></joint>'
            )
        tmp_urdf = "/tmp/_ur.urdf"
        with open(tmp_urdf, "w") as f:
            f.write(xml)

        self.chain = Chain.from_urdf_file(tmp_urdf, base_elements=["base_link"], active_links_mask=None)

        link_names = [getattr(l, "name", f"link_{i}") for i, l in enumerate(self.chain.links)]
        name_to_idx = {n: i for i, n in enumerate(link_names)}
        missing = [n for n in JOINT_NAMES if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                "Could not find these UR joints in URDF: "
                + ", ".join(missing)
                + f"\nAvailable: {link_names}"
            )
        self.active_idx = [name_to_idx[n] for n in JOINT_NAMES]
        self.q_full = np.zeros(len(self.chain.links))

    def solve(self, T_target: np.ndarray, q_seed: Optional[np.ndarray], mode_hint: str = "auto") -> np.ndarray:
        """
        Solve IK for the UR EE.
          mode_hint:
            - "all"      -> full 6D (position+orientation)
            - "Z"        -> constrain only tool Z-axis direction
            - "position" -> position-only
            - "auto"     -> try "all" -> "Z" -> "position"
        """
        if q_seed is None:
            q_seed = NOMINAL_Q

        # Warm start near nominal pose, favor small joint change
        q0 = self.q_full.copy()
        bseed = 0.9 * q_seed + 0.1 * NOMINAL_Q   # slightly stronger toward previous
        for k, idx in enumerate(self.active_idx):
            q0[idx] = bseed[k]

        def _extract(sol):
            return np.array([sol[idx] for idx in self.active_idx])

        def _ik_all():
            sol = self.chain.inverse_kinematics_frame(T_target, initial_position=q0, orientation_mode="all")
            return _extract(sol)

        def _ik_Z():
            sol = self.chain.inverse_kinematics_frame(T_target, initial_position=q0, orientation_mode="Z")
            return _extract(sol)

        def _ik_pos():
            sol = self.chain.inverse_kinematics(T_target[:3, 3], initial_position=q0)
            return _extract(sol)

        try:
            if mode_hint == "all":
                q = _ik_all()
            elif mode_hint == "Z":
                q = _ik_Z()
            elif mode_hint == "position":
                q = _ik_pos()
            else:
                # auto
                try:
                    q = _ik_all()
                except Exception:
                    try:
                        q = _ik_Z()
                    except Exception:
                        q = _ik_pos()
        except Exception:
            q = _ik_pos()

        q = clamp_soft(q)
        for k, idx in enumerate(self.active_idx):
            self.q_full[idx] = q[k]
        return q


# ----------------- CoppeliaSim driver -----------------
class CoppeliaDriver:
    def __init__(self, host: str, port: int, robot_base_path: str, joint_names: List[str]):
        self.client = RemoteAPIClient(host, port)
        self.sim = self.client.require("sim")

        # Ensure simulation runs; use stepped to synchronize streaming
        try:
            if hasattr(self.sim, "setStepping"):
                self.sim.setStepping(True)
        except Exception:
            pass
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        # Resolve base
        self.robot_path_used = self._resolve_path(robot_base_path)
        if self.robot_path_used is None:
            raise RuntimeError(f"Robot base not found at '{robot_base_path}' (tried aliases).")
        self.base_h = self.sim.getObject(self.robot_path_used)

        # Resolve joints
        self.joint_handles = []
        missing = []
        for jn in joint_names:
            h = None
            for cand in (f"{self.robot_path_used}/{jn}", jn, f"{jn}#0"):
                try:
                    h = self.sim.getObject(cand); break
                except Exception:
                    pass
            if h is None:
                missing.append(jn)
            else:
                self.joint_handles.append(h)
        if missing:
            raise RuntimeError("Missing joint(s) under robot base: " + ", ".join(missing))

        # Optional tool joint
        try:
            self.tool_h = self.sim.getObject(f"{self.robot_path_used}/tool_opening_joint")
        except Exception:
            self.tool_h = None

    def _resolve_path(self, path: str) -> Optional[str]:
        cands = []
        if path:
            cands += [path, path.rstrip("/"), path.lstrip("/")]
            if not path.endswith("#0"):
                cands.append(path + "#0")
        for cand in cands:
            try:
                self.sim.getObject(cand)
                return cand
            except Exception:
                continue
        return None

    def get_joints(self) -> np.ndarray:
        return np.array([self.sim.getJointPosition(h) for h in self.joint_handles], dtype=float)

    def set_tool(self, opening: float):
        if self.tool_h is not None:
            self.sim.setJointTargetPosition(self.tool_h, float(opening))

    def goto(self, q_target: np.ndarray, step_rad: float = LINEAR_STEP_RAD, settle: float = SETTLE_TIME_SEC):
        q_curr = self.get_joints()
        q_target = wrap_to_nearest(q_target, q_curr)
        q_target = clamp_soft(q_target)
        delta = q_target - q_curr
        steps = max(1, int(np.max(np.abs(delta)) / max(step_rad, 1e-5)))
        for s in range(1, steps + 1):
            q = q_curr + (s / steps) * delta
            for h, qi in zip(self.joint_handles, q):
                self.sim.setJointTargetPosition(h, float(qi))
            try:
                self.sim.step()
            except Exception:
                pass
        time.sleep(settle)


# ----------------- Planner for entry-only motion -----------------
def plan_entry_triplets(polyline_mat_m: np.ndarray,
                        spacing_m: float,
                        approach_z: float,
                        depth_m: float,
                        tilt_rad: float = POSE_TILT_RAD) -> List[PoseRPY]:
    """
    For each sampled point along the polyline (in MAT frame, meters),
    produce 3 poses: approach (z=+approach), pierce (z=-depth), retract (z=+approach).
    Orientation: tool Z down, yaw aligned with local tangent, small pitch tilt.
    """
    P = sample_polyline(polyline_mat_m, spacing_m)
    if len(P) == 0:
        return []

    poses: List[PoseRPY] = []
    z0 = 0.0  # mat surface at z=0 in MAT frame

    for i, p in enumerate(P[:, :2]):
        # tangent in XY
        if i == 0:
            t = P[min(1, len(P)-1), :2] - P[0, :2]
        elif i == len(P)-1:
            t = P[-1, :2] - P[-2, :2]
        else:
            t = P[i+1, :2] - P[i-1, :2]
        nrm = np.linalg.norm(t)
        if nrm < 1e-9:
            yaw = 0.0
        else:
            yaw = math.atan2(t[1], t[0])

        rpy = (math.pi, -float(tilt_rad), float(yaw))  # tool Z ~ down

        x = float(P[i, 0])
        y = float(P[i, 1])

        entry = np.array([x, y, z0 - float(depth_m)], dtype=float)
        appr  = np.array([x, y, z0 + float(approach_z)], dtype=float)

        poses.append(PoseRPY(appr,  rpy))  # approach
        poses.append(PoseRPY(entry, rpy))  # pierce
        poses.append(PoseRPY(appr,  rpy))  # retract

    return poses


# ----------------- Main ROS2 node -----------------
class EntryMotionNode(Node):
    def __init__(self):
        super().__init__("suture_entry_motion")

        # ---- Parameters ----
        self.declare_parameter("csim_host", CSIM_HOST_DEFAULT)
        self.declare_parameter("csim_port", CSIM_PORT_DEFAULT)
        self.declare_parameter("robot_base_path", ROBOT_BASE_PATH_DEFAULT)
        self.declare_parameter("suture_frame_path", SUTURE_FRAME_PATH_DEFAULT)
        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("approach_z", APPROACH_Z_DEFAULT)
        self.declare_parameter("default_depth", DEPTH_DEFAULT)
        self.declare_parameter("ur_type", "ur3")

        csim_host = self.get_parameter("csim_host").get_parameter_value().string_value
        csim_port = int(self.get_parameter("csim_port").get_parameter_value().integer_value or CSIM_PORT_DEFAULT)
        robot_base_path = self.get_parameter("robot_base_path").get_parameter_value().string_value
        suture_frame_path = self.get_parameter("suture_frame_path").get_parameter_value().string_value
        joint_names = list(self.get_parameter("joint_names").get_parameter_value().string_array_value or JOINT_NAMES)
        self.approach_z = float(self.get_parameter("approach_z").get_parameter_value().double_value or APPROACH_Z_DEFAULT)
        self.default_depth = float(self.get_parameter("default_depth").get_parameter_value().double_value or DEPTH_DEFAULT)
        ur_type = self.get_parameter("ur_type").get_parameter_value().string_value or "ur3"

        # ---- Connect to CoppeliaSim & robot ----
        self.driver = CoppeliaDriver(csim_host, csim_port, robot_base_path, joint_names)
        self.ik = IKUR(ur_type)

        # ---- Compute T_base_mat from scene ----
        def _resolve_handle(sim, path: str) -> Optional[int]:
            for cand in (path, path.rstrip("/"), path.lstrip("/"), path + "#0"):
                try:
                    return sim.getObject(cand)
                except Exception:
                    pass
            return None

        self.T_base_mat = np.eye(4)
        mat_h = _resolve_handle(self.driver.sim, suture_frame_path)
        if mat_h is None:
            raise RuntimeError(f"Cannot find '{suture_frame_path}' in CoppeliaSim scene.")
        pos = self.driver.sim.getObjectPosition(mat_h, self.driver.base_h)      # [x,y,z]
        rpy = self.driver.sim.getObjectOrientation(mat_h, self.driver.base_h)   # [r,p,y]
        R = tft.euler_matrix(*rpy)[:3, :3]
        self.T_base_mat[:3, :3] = R
        self.T_base_mat[:3,  3] = np.array(pos, dtype=float)

        d_reach = float(np.linalg.norm(self.T_base_mat[:3, 3]))
        self.get_logger().info(
            f"T_base_mat (from sim). pos={np.round(pos,6).tolist()}, rpy={np.round(rpy,6).tolist()}, "
            f"distance_from_base={d_reach:.3f} m"
        )

        # ---- ROS I/O ----
        self.sub = self.create_subscription(MsgString, "/suture_cuts", self.on_cuts, 10)
        self.get_logger().info("EntryMotionNode ready. Waiting for /suture_cuts ...")

    def on_cuts(self, msg: MsgString):
        """
        Execute stitched entry motions on the mat with a FIXED tool orientation
        (tool Z into the mat). Expects /suture_cuts JSON in meters, mat-centered XY:
        {
            "frame_id": "mat",
            "cuts": [ { "polyline": [[x,y], ...] }, ... ],
            "params": { "spacing": 0.008, "entry_mm": 0.006 (meters), "depth": 0.003 }
        }
        """
        # ---------- Parse ----------
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"/suture_cuts JSON parse error: {e}")
            return

        spacing   = float(data.get("params", {}).get("spacing", 0.008))
        approach  = float(data.get("params", {}).get("entry_mm", self.approach_z))  # already meters
        depth     = float(data.get("params", {}).get("depth",     self.default_depth))

        self.get_logger().info(
            f"/suture_cuts: {len(data.get('cuts', []))} cut(s) "
            f"(spacing={spacing:.4f} m, approach={approach:.4f} m, depth={depth:.4f} m)"
        )

        # ---------- Build a single, fixed tool orientation in BASE frame ----------
        Rbm = self.T_base_mat[:3, :3]  # mat axes in base
        tbm = self.T_base_mat[:3,  3]

        # tool Z points *into* the mat: -Z_mat in base coords
        z_tool_b = -Rbm[:, 2]
        z_tool_b /= (np.linalg.norm(z_tool_b) + 1e-12)

        # prefer tool X along mat +X (projected orthogonal to z_tool)
        x_pref_b = Rbm[:, 0]
        x_b = x_pref_b - np.dot(x_pref_b, z_tool_b) * z_tool_b
        if np.linalg.norm(x_b) < 1e-8:
            # degenerate; pick any axis not collinear with z_tool
            x_b = np.array([1.0, 0.0, 0.0], float)
            if abs(np.dot(x_b, z_tool_b)) > 0.9:
                x_b = np.array([0.0, 1.0, 0.0], float)
            x_b = x_b - np.dot(x_b, z_tool_b) * z_tool_b
        x_b /= (np.linalg.norm(x_b) + 1e-12)

        y_b = np.cross(z_tool_b, x_b); y_b /= (np.linalg.norm(y_b) + 1e-12)
        # columns are tool axes in base
        R_tool_b = np.column_stack([x_b, y_b, z_tool_b])

        def T_from_R_p(R: np.ndarray, p: np.ndarray) -> np.ndarray:
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T

        # ---------- Actuate ----------
        self.driver.set_tool(0.02)  # slightly open tool
        q_seed = self.driver.get_joints()

        for cut in data.get("cuts", []):
            poly = np.array(cut.get("polyline", []), dtype=float)
            if poly.ndim != 2 or poly.shape[1] < 2 or len(poly) < 2:
                self.get_logger().warn("Skipping malformed polyline"); continue

            # clamp to pad footprint (centered; 1 mm margin)
            poly[:, 0] = np.clip(poly[:, 0], -MAT_HALF_X + 1e-3, MAT_HALF_X - 1e-3)
            poly[:, 1] = np.clip(poly[:, 1], -MAT_HALF_Y + 1e-3, MAT_HALF_Y - 1e-3)

            # resample for smoother motion (keep z=0 in mat frame; we add offsets)
            P = sample_polyline(np.c_[poly[:, :2], np.zeros(len(poly))], spacing)
            if P.shape[0] == 0:
                self.get_logger().warn("Empty path after resampling"); continue

            self.get_logger().info(f"Executing {P.shape[0]} points with triplets (approach/pierce/retract).")

            prev_xy = None
            for i in range(P.shape[0]):
                xy = P[i, :2].astype(float)

                # three Z levels relative to mat surface
                z_levels = [
                    (+abs(approach), "approach"),
                    (-abs(depth),    "pierce"),
                    (+abs(approach), "retract"),
                ]

                for z_off, phase in z_levels:
                    # 1) Z-only at prev_xy (keep XY fixed), **full orientation locked**
                    if prev_xy is None:
                        prev_xy = xy.copy()

                    p_mat = np.array([prev_xy[0], prev_xy[1], z_off], float)
                    p_base = Rbm @ p_mat + tbm
                    Tz = T_from_R_p(R_tool_b, p_base)

                    try:
                        qA = self.ik.solve(Tz, q_seed, mode_hint="all")  # full 6D; prevents wrist spin
                        # light wrist-3 damping to discourage micro-spin
                        qA = wrap_to_nearest(qA, q_seed)
                        qA[5] = 0.8 * q_seed[5] + 0.2 * qA[5]
                        qA = clamp_soft(qA)
                        self.driver.goto(qA)
                        q_seed = qA.copy()
                    except Exception as e:
                        self.get_logger().warn(f"{phase} Z-only failed at point {i}: {e}")

                    # 2) XY-at-constant-Z to target xy, **same fixed orientation**
                    p_mat = np.array([xy[0], xy[1], z_off], float)
                    p_base = Rbm @ p_mat + tbm
                    Txy = T_from_R_p(R_tool_b, p_base)

                    try:
                        qB = self.ik.solve(Txy, q_seed, mode_hint="all")
                        qB = wrap_to_nearest(qB, q_seed)
                        qB[5] = 0.8 * q_seed[5] + 0.2 * qB[5]
                        qB = clamp_soft(qB)
                        self.driver.goto(qB)
                        q_seed = qB.copy()
                        prev_xy = xy.copy()
                    except Exception as e:
                        self.get_logger().warn(f"{phase} XY-at-Z failed at point {i}: {e}")

        self.driver.set_tool(0.02)
        self.get_logger().info("Finished executing entry motions.")
        

def main():
    rclpy.init()
    node = None
    try:
        node = EntryMotionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
