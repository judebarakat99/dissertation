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
from ament_index_python.packages import get_package_share_directory

# CoppeliaSim ZMQ Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


# ----------------- Defaults / Tunables -----------------
CSIM_HOST_DEFAULT = "127.0.0.1"
CSIM_PORT_DEFAULT = 23000

ROBOT_BASE_PATH_DEFAULT = "/UR3"     # your UR3 root in the scene
SUTURE_FRAME_PATH_DEFAULT = "/mat"   # your dummy 'mat' (parent of suture_pad[1])

JOINT_NAMES_DEFAULT = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Approach / motion shaping (meters / radians)
APPROACH_Z_DEFAULT = 0.012          # 12 mm above surface
DEPTH_DEFAULT      = 0.003          # 3 mm down into mat (pierce)
LINEAR_STEP_RAD    = 0.01           # max joint change per streamed step
SETTLE_TIME_SEC    = 0.004
POSE_TILT_RAD      = 0.20           # ~11.5°, small pitch to avoid straight wrist

# Preferred seed posture (UR3 elbow-up-ish)
NOMINAL_Q = np.array([0.0, -1.35, 1.90, 0.0, 1.30, 0.0])

# Soft joint limits (adjust if your scene needs more freedom)
#          J1 (pan)         J2 (shoulder)   J3 (elbow)      J4 (wrist1)   J5 (wrist2)   J6 (wrist3)
SOFT_LIMITS = np.array([
    [-math.pi,  math.pi],   # [-180°, 180°]
    [-2.60,     -0.10],
    [ 0.00,      3.00],
    [-2.50,      2.50],
    [ 0.25,      2.80],     # keep away from 0 (straight wrist) and extremes
    [-math.pi,  math.pi],
], dtype=float)
TWOPI = 2.0 * math.pi
# -------------------------------------------------------


@dataclass
class PoseRPY:
    xyz: np.ndarray                 # (3,)
    rpy: Tuple[float, float, float] # roll, pitch, yaw


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

        target_joint_names = JOINT_NAMES_DEFAULT
        link_names = [getattr(l, "name", f"link_{i}") for i, l in enumerate(self.chain.links)]
        name_to_idx = {n: i for i, n in enumerate(link_names)}
        missing = [n for n in target_joint_names if n not in name_to_idx]
        if missing:
            raise RuntimeError(
                "Could not find these UR joints in URDF: "
                + ", ".join(missing)
                + f"\nAvailable: {link_names}"
            )
        self.active_idx = [name_to_idx[n] for n in target_joint_names]
        self.q_full = np.zeros(len(self.chain.links))

    def solve(self, T_target: np.ndarray, q_seed: Optional[np.ndarray]) -> np.ndarray:
        if q_seed is None:
            q_seed = NOMINAL_Q
        # Warm start near nominal pose
        q0 = self.q_full.copy()
        bseed = blend_seed(q_seed)
        for k, idx in enumerate(self.active_idx):
            q0[idx] = bseed[k]

        # Try 6D orientation first, fallback to Z-only
        try:
            sol_all = self.chain.inverse_kinematics_frame(T_target, initial_position=q0, orientation_mode="all")
            q = np.array([sol_all[idx] for idx in self.active_idx])
        except Exception:
            sol_z = self.chain.inverse_kinematics_frame(T_target, initial_position=q0, orientation_mode="z")
            q = np.array([sol_z[idx] for idx in self.active_idx])

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

        rpy = (math.pi, -float(tilt_rad), float(yaw))

        entry = np.array([P[i, 0], P[i, 1], z0 - float(depth_m)], dtype=float)
        appr  = np.array([P[i, 0], P[i, 1], z0 + float(approach_z)], dtype=float)

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
        self.declare_parameter("joint_names", JOINT_NAMES_DEFAULT)
        self.declare_parameter("approach_z", APPROACH_Z_DEFAULT)
        self.declare_parameter("default_depth", DEPTH_DEFAULT)
        self.declare_parameter("ur_type", "ur3")

        csim_host = self.get_parameter("csim_host").get_parameter_value().string_value
        csim_port = int(self.get_parameter("csim_port").get_parameter_value().integer_value or CSIM_PORT_DEFAULT)
        robot_base_path = self.get_parameter("robot_base_path").get_parameter_value().string_value
        suture_frame_path = self.get_parameter("suture_frame_path").get_parameter_value().string_value
        joint_names = list(self.get_parameter("joint_names").get_parameter_value().string_array_value or JOINT_NAMES_DEFAULT)
        self.approach_z = float(self.get_parameter("approach_z").get_parameter_value().double_value or APPROACH_Z_DEFAULT)
        self.default_depth = float(self.get_parameter("default_depth").get_parameter_value().double_value or DEPTH_DEFAULT)
        ur_type = self.get_parameter("ur_type").get_parameter_value().string_value or "ur3"

        # ---- Connect to CoppeliaSim & robot ----
        self.driver = CoppeliaDriver(csim_host, csim_port, robot_base_path, joint_names)
        self.ik = IKUR(ur_type)

        # ---- Compute T_base_mat from scene ----
        self.T_base_mat = np.eye(4)
        self._mat_handle = self._resolve_handle(self.driver.sim, suture_frame_path)
        if self._mat_handle is None:
            raise RuntimeError(f"Cannot find '{suture_frame_path}' in CoppeliaSim scene.")
        pos = self.driver.sim.getObjectPosition(self._mat_handle, self.driver.base_h)      # [x,y,z]
        rpy = self.driver.sim.getObjectOrientation(self._mat_handle, self.driver.base_h)   # [r,p,y]
        R = tft.euler_matrix(*rpy)[:3, :3]
        self.T_base_mat[:3, :3] = R
        self.T_base_mat[:3,  3] = np.array(pos, dtype=float)

        self.get_logger().info(
            f"T_base_mat set from scene. frame='{suture_frame_path}' "
            f"pos={np.round(pos,6).tolist()}, rpy={np.round(rpy,6).tolist()}"
        )

        # ---- ROS I/O ----
        self.sub = self.create_subscription(MsgString, "/suture_cuts", self.on_cuts, 10)
        self.get_logger().info("EntryMotionNode ready. Waiting for /suture_cuts ...")

    # Robust resolver for scene objects
    @staticmethod
    def _resolve_handle(sim, path: str) -> Optional[int]:
        cands = []
        if path:
            cands += [path, path.rstrip("/"), path.lstrip("/")]
            if not path.endswith("#0"):
                cands.append(path + "#0")
        for cand in cands:
            try:
                return sim.getObject(cand)
            except Exception:
                continue
        return None

    def on_cuts(self, msg: MsgString):
        """
        Expected JSON (meters):
        {
          "frame_id": "mat",
          "cuts": [ { "polyline": [[x,y], ...] }, ... ],
          "params": { "spacing": 0.008, "entry_mm": 0.006 (m!), "depth": 0.003 }
        }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"/suture_cuts JSON parse error: {e}")
            return

        spacing = float(data.get("params", {}).get("spacing", 0.008))
        entry_m = float(data.get("params", {}).get("entry_mm", 0.006))  # already in meters (from vision_web)
        depth   = float(data.get("params", {}).get("depth",   self.default_depth))

        self.get_logger().info(
            f"/suture_cuts: {len(data.get('cuts', []))} cuts (spacing={spacing:.3f} m, "
            f"entry={entry_m:.3f} m, depth={depth:.3f} m)"
        )

        # Slightly open tool if present
        self.driver.set_tool(0.02)

        q_seed = self.driver.get_joints()

        for cut in data.get("cuts", []):
            poly = np.array(cut.get("polyline", []), dtype=float)
            if poly.ndim != 2 or poly.shape[1] not in (2, 3) or len(poly) < 2:
                self.get_logger().warn("Skipping malformed polyline")
                continue

            # Plan simple entry triplets in MAT frame
            poses = plan_entry_triplets(poly, spacing, self.approach_z, depth, POSE_TILT_RAD)
            self.get_logger().info(f"Planned {len(poses)} poses.")

            # Execute
            for k, pose in enumerate(poses):
                T_mat = rpy_to_T(pose.xyz, pose.rpy)
                T_base = self.T_base_mat @ T_mat
                try:
                    q_target = self.ik.solve(T_base, q_seed)
                    q_target = wrap_to_nearest(q_target, q_seed)
                    q_target = clamp_soft(q_target)
                    self.driver.goto(q_target)
                    q_seed = q_target.copy()
                except Exception as e:
                    self.get_logger().warn(f"IK/exec failed at step {k}: {e}")

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
