#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, json, subprocess
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as MsgString
from ikpy.chain import Chain

# CoppeliaSim ZMQ Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


# ====================== Tunables ======================

CSIM_HOST_DEFAULT = "127.0.0.1"
CSIM_PORT_DEFAULT = 23000

ROBOT_BASE_PATH_DEFAULT = "/UR3"   # UR3 root in scene
SUTURE_FRAME_PATH_DEFAULT = "/mat" # mat dummy

JOINT_NAMES = [
    "shoulder_pan_joint",  # J1
    "shoulder_lift_joint", # J2
    "elbow_joint",         # J3
    "wrist_1_joint",       # J4
    "wrist_2_joint",       # J5
    "wrist_3_joint",       # J6
]

# Mat physical size (meters) — centered origin in /mat
MAT_HALF_X = 0.2975 * 0.5   # 0.14875 m
MAT_HALF_Y = 0.4250  * 0.5  # 0.21250 m

# Approach / motion shaping
APPROACH_Z_DEFAULT = 0.012      # 12 mm above surface (in /mat z)
DEPTH_DEFAULT      = 0.003      # 3 mm below surface (pierce)
LINEAR_STEP_RAD    = 0.01
SETTLE_TIME_SEC    = 0.004

# Joint soft limits
SOFT_LIMITS = np.array([
    [-math.pi,  math.pi],  # J1
    [-2.60,     -0.10],    # J2
    [ 0.00,      3.00],    # J3
    [-2.50,      2.50],    # J4
    [ 0.25,      2.80],    # J5
    [-2.20,      2.20],    # J6
], dtype=float)
TWOPI = 2.0 * math.pi

# DLS IK params (J1..J3 only)
DLS_LAMBDA   = 0.02      # damping
DLS_GAIN     = 0.8       # step gain
MAX_DQ_ARM   = 0.10      # max per-iter change for J1..J3 (rad)
POS_TOL      = 5e-4      # 0.5 mm tolerance
MAX_ITERS    = 120
JAC_EPS      = 1e-4      # rad, numeric jacobian epsilon


# ====================== Small helpers ======================

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

def clamp_xy_to_mat(xy: np.ndarray) -> np.ndarray:
    """Clamp XY to the centered mat rectangle with 1 mm margin."""
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

def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),     2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], dtype=float)

def pose7_to_T(pose7):
    x, y, z, qx, qy, qz, qw = pose7
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3,  3] = [x, y, z]
    return T


# ====================== CoppeliaSim driver ======================

class CoppeliaDriver:
    def __init__(self, host: str, port: int, robot_base_path: str, joint_names: List[str]):
        self.client = RemoteAPIClient(host, port)
        self.sim = self.client.require("sim")

        # Stepped sim for smooth streaming
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


# ====================== DLS (J1..J3 only, wrist locked) ======================

class J13DLS:
    """
    Position-only DLS IK using only J1..J3 (shoulder pan/lift, elbow).
    J4..J6 are held constant -> NO WRIST SPIN.
    """
    def __init__(self, chain: Chain, active_idx: List[int]):
        self.chain = chain
        self.active_idx = active_idx      # 6 entries mapping J1..J6 to ikpy link indices
        self.q_full = np.zeros(len(self.chain.links))

    def _fk_pos(self, q6: np.ndarray) -> np.ndarray:
        q_full = self.q_full.copy()
        for k, idx in enumerate(self.active_idx):
            q_full[idx] = q6[k]
        T = self.chain.forward_kinematics(q_full)
        return T[:3, 3].astype(float)

    def _num_jac_pos_j13(self, q6: np.ndarray, eps: float) -> np.ndarray:
        """
        Numeric Jacobian (3x3) wrt joints 0,1,2 only. Columns are d p / d qj.
        """
        J = np.zeros((3, 3), dtype=float)
        for j in range(3):  # J1..J3 only
            dq = np.zeros(6); dq[j] = eps
            p_plus  = self._fk_pos(q6 + dq)
            p_minus = self._fk_pos(q6 - dq)
            J[:, j] = (p_plus - p_minus) / (2.0 * eps)
        return J

    def solve(self,
              p_target_base: np.ndarray,
              q_seed: np.ndarray,
              max_iters: int = MAX_ITERS,
              tol: float = POS_TOL,
              lam: float = DLS_LAMBDA,
              gain: float = DLS_GAIN,
              max_dq: float = MAX_DQ_ARM) -> np.ndarray:
        """
        Drive TCP position to target with J1..J3 only. J4..J6 remain exactly as in q_seed.
        """
        q = q_seed.copy()
        # Freeze current wrist
        q[3:] = q_seed[3:]
        for _ in range(max_iters):
            p_cur = self._fk_pos(q)
            e = p_target_base - p_cur
            if float(np.linalg.norm(e)) < tol:
                break

            J = self._num_jac_pos_j13(q, JAC_EPS)  # 3x3

            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            rhs = gain * e
            try:
                v = np.linalg.solve(JJt, rhs)   # 3x1
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(JJt, rhs, rcond=None)[0]
            dq13 = J.T @ v                      # 3x1

            dq13 = np.clip(dq13, -max_dq, max_dq)

            q[:3] = q[:3] + dq13
            q = clamp_soft(q)

        # ensure wrist preserved exactly
        q[3:] = q_seed[3:]
        return q


# ====================== Node ======================

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

        # ---- CoppeliaSim ----
        self.driver = CoppeliaDriver(csim_host, csim_port, robot_base_path, joint_names)

        # ---- T_base_mat from quaternion pose ----
        def _resolve_handle(sim, path: str) -> Optional[int]:
            for cand in (path, path.rstrip("/"), path.lstrip("/"), path + "#0"):
                try:
                    return sim.getObject(cand)
                except Exception:
                    pass
            return None
        mat_h = _resolve_handle(self.driver.sim, suture_frame_path)
        if mat_h is None:
            raise RuntimeError(f"Cannot find '{suture_frame_path}' in scene.")
        pose7 = self.driver.sim.getObjectPose(mat_h, self.driver.base_h)  # [x,y,z,qx,qy,qz,qw]
        self.T_base_mat = pose7_to_T(pose7)
        pos = self.T_base_mat[:3, 3]
        self.get_logger().info(f"T_base_mat pos={np.round(pos,6).tolist()}, distance_from_base={float(np.linalg.norm(pos)):.3f} m")

        # ---- Build IKPy chain (for FK only) ----
        from ament_index_python.packages import get_package_share_directory
        ur_desc = get_package_share_directory("ur_description")
        xacro_path = os.path.join(ur_desc, "urdf", "ur.urdf.xacro")
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
            raise RuntimeError(f"URDF missing joints: {', '.join(missing)}\nAvailable: {link_names}")
        self.active_idx = [name_to_idx[n] for n in JOINT_NAMES]
        self.dls = J13DLS(self.chain, self.active_idx)

        # ---- ROS I/O ----
        self.sub = self.create_subscription(MsgString, "/suture_cuts", self.on_cuts, 10)
        self.get_logger().info("EntryMotionNode ready. Waiting for /suture_cuts ...")

    # ----------------- /suture_cuts callback -----------------
    def on_cuts(self, msg: MsgString):
        """
        Execute stitched entry motions with position-only DLS on J1..J3 ONLY.
        J4..J6 remain fixed => no wrist spin.
        """
        # Parse
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"/suture_cuts JSON parse error: {e}")
            return

        spacing  = float(data.get("params", {}).get("spacing", 0.008))
        approach = float(data.get("params", {}).get("entry_mm", self.approach_z))  # already meters
        depth    = float(data.get("params", {}).get("depth",     self.default_depth))

        self.get_logger().info(
            f"/suture_cuts: {len(data.get('cuts', []))} cut(s) "
            f"(spacing={spacing:.4f} m, approach={approach:.4f} m, depth={depth:.4f} m)"
        )

        self.driver.set_tool(0.02)
        q_seed = self.driver.get_joints()

        Rbm = self.T_base_mat[:3, :3]
        tbm = self.T_base_mat[:3,  3]

        for cut in data.get("cuts", []):
            poly = np.array(cut.get("polyline", []), dtype=float)
            if poly.ndim != 2 or poly.shape[1] < 2 or len(poly) < 2:
                self.get_logger().warn("Skipping malformed polyline"); continue

            # Clamp to pad footprint
            poly[:, 0] = np.clip(poly[:, 0], -MAT_HALF_X + 1e-3, MAT_HALF_X - 1e-3)
            poly[:, 1] = np.clip(poly[:, 1], -MAT_HALF_Y + 1e-3, MAT_HALF_Y - 1e-3)

            # Resample
            P = sample_polyline(np.c_[poly[:, :2], np.zeros(len(poly))], spacing)
            if P.shape[0] == 0:
                self.get_logger().warn("Empty path after resampling"); continue

            self.get_logger().info(f"Executing {P.shape[0]} points (each → approach / pierce / retract).")

            prev_xy = None
            for i in range(P.shape[0]):
                xy = clamp_xy_to_mat(P[i, :2])

                # Z levels relative to mat
                z_levels = [
                    (+abs(approach), "approach"),
                    (-abs(depth),    "pierce"),
                    (+abs(approach), "retract"),
                ]

                for z_off, phase in z_levels:
                    # 1) Z-only at prev_xy (keep XY fixed)
                    if prev_xy is None:
                        prev_xy = xy.copy()

                    p_mat  = np.array([prev_xy[0], prev_xy[1], z_off], float)
                    p_base = Rbm @ p_mat + tbm

                    try:
                        qA = self.dls.solve(
                            p_target_base=p_base,
                            q_seed=q_seed,
                            max_iters=MAX_ITERS, tol=POS_TOL,
                            lam=DLS_LAMBDA, gain=DLS_GAIN,
                            max_dq=MAX_DQ_ARM
                        )
                        qA = wrap_to_nearest(qA, q_seed); qA = clamp_soft(qA)
                        self.driver.goto(qA)
                        q_seed = qA.copy()
                    except Exception as e:
                        self.get_logger().warn(f"{phase} Z-only failed at point {i}: {e}")

                    # 2) XY-at-constant-Z to target xy
                    p_mat  = np.array([xy[0], xy[1], z_off], float)
                    p_base = Rbm @ p_mat + tbm

                    try:
                        qB = self.dls.solve(
                            p_target_base=p_base,
                            q_seed=q_seed,
                            max_iters=MAX_ITERS, tol=POS_TOL,
                            lam=DLS_LAMBDA, gain=DLS_GAIN,
                            max_dq=MAX_DQ_ARM
                        )
                        qB = wrap_to_nearest(qB, q_seed); qB = clamp_soft(qB)
                        self.driver.goto(qB)
                        q_seed = qB.copy()
                        prev_xy = xy.copy()
                    except Exception as e:
                        self.get_logger().warn(f"{phase} XY-at-Z failed at point {i}: {e}")

        self.driver.set_tool(0.02)
        self.get_logger().info("Finished executing entry motions.")


# ====================== main ======================

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
