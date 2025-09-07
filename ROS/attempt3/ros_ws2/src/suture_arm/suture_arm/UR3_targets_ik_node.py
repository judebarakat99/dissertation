#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR3 servo in CoppeliaSim using built-in simIK (DLS) with tilt-only orientation.

- Uses CoppeliaSim's simIK (damped least squares).
- Position + alphaBeta (tilt) constraints (yaw free → no endless spinning).
- Respects joint limits/topology; applies solutions back to the scene each tick.
"""

import os
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf_transformations as tft

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy

# ---------------- Config ----------------
CSIM_HOST = os.getenv("CSIM_HOST", "127.0.0.1")
CSIM_PORT = int(os.getenv("CSIM_PORT", "23000"))

ROBOT_NAME_IN_SCENE = "UR3"
ROBOT_BASE_PATH = f"/{ROBOT_NAME_IN_SCENE}"
UR3_TARGET_PATHS = [f"/{ROBOT_NAME_IN_SCENE}/UR3_target", "/UR3_target", "UR3_target", "UR3_target#0"]
MOVING_TARGET_PATHS = ["/moving_target", "moving_target", "/moving_target#0", "moving_target#0"]

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ---------- Defaults ----------
DEFAULT_DOWN_RPY = (math.pi, 0.0, 0.0)   # tool-z down
DEFAULT_WORK_Z   = float(os.getenv("WORK_Z", "0.150"))

# UR3 conservative reach (meters)
UR3_MAX_REACH_M = 0.49

# IK tuning
IK_DLS_DAMPING        = 0.01     # λ
IK_MAX_ITERS_PER_TICK = 3        # solver inner iters per sim step
IK_TICK_COUNT         = 1200     # hard cap per target
IK_POS_THR_M          = 0.0018   # ~1.8 mm
IK_TILT_THR_RAD       = math.radians(1.5)

PRINT_EVERY_STEPS = 15

# ------------- Helpers -------------
def to_pose_dict(pose_like) -> Dict:
    if isinstance(pose_like, (list, tuple)):
        if len(pose_like) == 3:
            return {"pos": [float(pose_like[0]), float(pose_like[1]), float(pose_like[2])], "rpy": None}
        if len(pose_like) >= 6:
            return {"pos": [float(pose_like[0]), float(pose_like[1]), float(pose_like[2])],
                    "rpy": (float(pose_like[3]), float(pose_like[4]), float(pose_like[5]))}
    if isinstance(pose_like, dict):
        if "pos" in pose_like:
            pos = pose_like["pos"]
            rpy = pose_like.get("rpy")
        else:
            pos = [pose_like.get("x", 0.0), pose_like.get("y", 0.0), pose_like.get("z", 0.0)]
            rpy = [pose_like.get("roll"), pose_like.get("pitch"), pose_like.get("yaw")]
            if any(v is None for v in rpy):
                rpy = None
        return {"pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "rpy": (tuple(float(v) for v in rpy) if rpy is not None else None)}
    raise ValueError(f"Unrecognized target format: {pose_like}")

def np_norm(v): return float(np.linalg.norm(v))

# ---------- sim wrapper ----------
class CoppeliaIK:
    def __init__(self, node: Node):
        self.node = node
        self.client = RemoteAPIClient(CSIM_HOST, CSIM_PORT)
        self.sim = self.client.require("sim")
        self.simIK = self.client.require("simIK")

        try:
            self.sim.setStepping(True)
        except Exception:
            pass

        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        self.base = self._resolve_one([ROBOT_BASE_PATH, f"{ROBOT_BASE_PATH}#0", ROBOT_NAME_IN_SCENE, f"{ROBOT_NAME_IN_SCENE}#0"])
        if self.base is None:
            raise RuntimeError(f"Robot base '{ROBOT_BASE_PATH}' not found.")

        # collect joints
        self.joints = []
        for jn in JOINT_NAMES:
            h = self._resolve_one([f"{ROBOT_BASE_PATH}/{jn}", jn, f"{jn}#0"])
            if h is None:
                raise RuntimeError(f"Could not find joint {jn}")
            self.joints.append(h)

        self.tip = self._resolve_one([f"{ROBOT_BASE_PATH}/UR3_tip", "/UR3_tip", "UR3_tip", "UR3_tip#0"])
        if self.tip is None:
            raise RuntimeError("UR3_tip not found.")

        self.ur3_target = self._resolve_one(UR3_TARGET_PATHS)
        self.moving_target = None if self.ur3_target else self._resolve_one(MOVING_TARGET_PATHS)

        if self.ur3_target:
            self.target = self.ur3_target
            self.target_parent = self.base
            node.get_logger().info("Using dummy: UR3_target (parent=UR3).")
        elif self.moving_target:
            self.target = self.moving_target
            self.target_parent = -1
            node.get_logger().info("Using dummy: moving_target (parent=scene).")
        else:
            raise RuntimeError("No target dummy found.")

        # timestep
        try:
            self.dt = float(self.sim.getSimulationTimeStep())
            if self.dt <= 0.0:
                self.dt = 0.05
        except Exception:
            self.dt = 0.05

        node.get_logger().info(f"Servo (simIK) enabled. dt={self.dt:.4f}s")

        # Build simIK environment & group once
        self._build_ik_env()

    def _resolve_one(self, cands: List[str]) -> Optional[int]:
        for c in cands:
            try:
                return self.sim.getObject(c)
            except Exception:
                pass
        return None

    def _build_ik_env(self):
        ik = self.simIK
        sim = self.sim

        self.ikEnv = ik.createEnvironment()
        self.ikGroup = ik.createGroup(self.ikEnv)

        # NOTE: simIK.setGroupCalculation requires the environment handle first.
        ik.setGroupCalculation(self.ikEnv, self.ikGroup, ik.method_damped_least_squares, IK_DLS_DAMPING, 50)

        # Constraints: position + alphaBeta (tilt). Intentionally ignore gamma (yaw).
        self.CONSTR_POS_TILT = ik.constraint_x | ik.constraint_y | ik.constraint_z | ik.constraint_alpha_beta

        # Map the current scene chain into IK (base→tip) with target
        self.ikElement = ik.addElementFromScene(self.ikEnv, self.ikGroup, self.base, self.tip, self.target, self.CONSTR_POS_TILT)

        # Make sure joints are in motorized pos-control; use setJointTargetForce (not deprecated setJointMaxForce)
        for j in self.joints:
            try:
                sim.setJointMode(j, sim.jointmode_force, 0)
            except Exception:
                pass
            try:
                sim.setObjectInt32Param(j, sim.jointintparam_ctrl_enabled, 1)
            except Exception:
                pass
            try:
                sim.setJointTargetVelocity(j, 4.0)
            except Exception:
                pass
            try:
                sim.setJointTargetForce(j, 120.0)  # replaces setJointMaxForce (deprecated)
            except Exception:
                pass

        for _ in range(3):
            sim.step()

    # ---- motion helpers ----
    def set_dummy_pose(self, pos_parent: List[float], rpy_parent: Optional[Tuple[float, float, float]]):
        self.sim.setObjectPosition(self.target, self.target_parent, list(map(float, pos_parent)))
        if rpy_parent is not None:
            self.sim.setObjectOrientation(self.target, self.target_parent, list(map(float, rpy_parent)))

    def get_tip_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.sim.getObjectPosition(self.tip, self.base), dtype=float)
        rpy = self.sim.getObjectOrientation(self.tip, self.base)
        R = tft.euler_matrix(*rpy)[:3, :3]
        return pos, R

    def get_target_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.sim.getObjectPosition(self.target, self.base), dtype=float)
        rpy = self.sim.getObjectOrientation(self.target, self.base)
        R = tft.euler_matrix(*rpy)[:3, :3]
        return pos, R

    def ik_servo_to_target(self, timeout_s: float = 240.0):
        ik = self.simIK
        sim = self.sim

        t0 = time.time()
        steps = 0
        opts = {"syncWorlds": True}  # pulls scene→IK, solves, pushes IK→scene

        while steps < IK_TICK_COUNT:
            # a few inner iterations per sim step for smooth motion
            for _ in range(IK_MAX_ITERS_PER_TICK):
                ik.handleGroup(self.ikEnv, self.ikGroup, opts)

            sim.step()
            steps += 1

            # check convergence (pos + tilt only)
            p_tip, R_tip = self.get_tip_pose_in_base()
            p_tgt, R_tgt = self.get_target_pose_in_base()
            pos_err = float(np.linalg.norm(p_tgt - p_tip))

            z_tip = R_tip[:, 2] / max(1e-9, np.linalg.norm(R_tip[:, 2]))
            z_tgt = R_tgt[:, 2] / max(1e-9, np.linalg.norm(R_tgt[:, 2]))
            tilt_err = math.acos(float(np.clip(np.dot(z_tip, z_tgt), -1.0, 1.0)))

            if steps % PRINT_EVERY_STEPS == 0:
                self.node.get_logger().info(
                    f"  IK tick {steps}: pos_err={pos_err*1000:.1f} mm, tilt_err={math.degrees(tilt_err):.1f} deg"
                )

            if pos_err < IK_POS_THR_M and tilt_err < IK_TILT_THR_RAD:
                break

            if (time.time() - t0) > timeout_s:
                self.node.get_logger().warn(
                    f"simIK servo timeout at tick {steps}: pos_err={pos_err*1000:.1f} mm, tilt_err={math.degrees(tilt_err):.1f} deg"
                )
                break

        for _ in range(3):
            sim.step()

    def home_all_joints_zero(self):
        # quick homing: set both current pos and targets to 0
        for h in self.joints:
            try:
                self.sim.setJointPosition(h, 0.0)
            except Exception:
                pass
            try:
                self.sim.setJointTargetPosition(h, 0.0)
            except Exception:
                pass
        for _ in range(10):
            self.sim.step()


# ---------- ROS Node ----------
class VisionTargetsIKNode(Node):
    def __init__(self):
        super().__init__("vision_targets_ik")

        # Parameters (overridable)
        self.declare_parameter("work_z", DEFAULT_WORK_Z)
        self.declare_parameter("down_rpy", list(DEFAULT_DOWN_RPY))
        self.declare_parameter("force_down_orientation", True)
        self.declare_parameter("force_work_z", True)
        self.declare_parameter("home_on_start", True)
        self.declare_parameter("ik_damping", IK_DLS_DAMPING)

        self.work_z          = float(self.get_parameter("work_z").value)
        self.down_rpy        = tuple(float(x) for x in self.get_parameter("down_rpy").value)
        self.force_down      = bool(self.get_parameter("force_down_orientation").value)
        self.force_work_z    = bool(self.get_parameter("force_work_z").value)
        self.home_on_start   = bool(self.get_parameter("home_on_start").value)
        damping_in           = float(self.get_parameter("ik_damping").value)

        self.get_logger().info(f"Connecting to CoppeliaSim at {CSIM_HOST}:{CSIM_PORT} ...")
        self.sim = CoppeliaIK(self)

        # allow damping override (NOTE: env handle required)
        try:
            self.sim.simIK.setGroupCalculation(self.sim.ikEnv, self.sim.ikGroup, self.sim.simIK.method_damped_least_squares, float(damping_in), 50)
            self.get_logger().info(f"simIK damping set to {float(damping_in)}")
        except Exception:
            pass

        # Homing
        if self.home_on_start:
            self.get_logger().info("Homing: setting all joints to 0 rad ...")
            self.sim.home_all_joints_zero()
            self.get_logger().info("Homing complete. Waiting on /vision_targets ...")
        else:
            self.get_logger().info("Startup homing disabled. Waiting on /vision_targets ...")

        self.sub = self.create_subscription(String, "/vision_targets", self.on_msg, 10)

    def on_msg(self, msg: String):
        self.get_logger().info(f"/vision_targets payload: {msg.data}")

        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}")
            return

        frame = (payload.get("frame") or payload.get("frame_id") or "scene").lower()
        targets_raw = payload.get("targets")
        if not targets_raw:
            self.get_logger().warn("No 'targets' in payload.")
            return

        for i, trg in enumerate(targets_raw):
            try:
                t = to_pose_dict(trg)
            except Exception as e:
                self.get_logger().warn(f"Skipping malformed target #{i}: {e}")
                continue

            # Plane & orientation
            pos = t["pos"]
            if self.force_work_z:
                pos[2] = self.work_z

            if self.force_down:
                rpy_in = self.down_rpy
                want_orientation = True
            else:
                want_orientation = t["rpy"] is not None
                if want_orientation:
                    rpy_in = t["rpy"]
                else:
                    rpy_in = self.sim.sim.getObjectOrientation(self.sim.target, self.sim.target_parent)

            # Transform into dummy's parent frame (scene or base)
            pos_parent, rpy_parent = self._to_parent_frame(pos, rpy_in, frame)

            # Workspace guard when target parent is UR3 base:
            if self.sim.target_parent == self.sim.base:
                r_xy = float(np.linalg.norm([pos_parent[0], pos_parent[1]]))
                if r_xy > (UR3_MAX_REACH_M - 0.02):
                    scale = (UR3_MAX_REACH_M - 0.02) / r_xy
                    pos_parent = [pos_parent[0]*scale, pos_parent[1]*scale, self.work_z]
                    self.get_logger().warn(
                        f"Target outside UR3 XY reach (r={r_xy:.3f} m > {UR3_MAX_REACH_M:.3f} m). "
                        f"Projecting XY to {np.round(pos_parent[:2],5).tolist()} and clamping Z={self.work_z:.3f}."
                    )
                else:
                    pos_parent = [pos_parent[0], pos_parent[1], self.work_z]

            self.get_logger().info(
                f"[{i+1}/{len(targets_raw)}] Move dummy → pos={np.round(pos_parent,5).tolist()}, "
                f"rpy={tuple(round(v,5) for v in rpy_parent)} (frame={frame}, tilt-only={want_orientation})"
            )
            self.sim.set_dummy_pose(pos_parent, rpy_parent if want_orientation else None)

            # Run simIK servo
            self.get_logger().info(f"[{i+1}/{len(targets_raw)}] simIK servo to dummy ...")
            t0 = time.time()
            self.sim.ik_servo_to_target(timeout_s=240.0)
            t_el = time.time() - t0

            p_tip, _ = self.sim.get_tip_pose_in_base()
            p_tgt, _ = self.sim.get_target_pose_in_base()
            pos_res = float(np.linalg.norm(p_tip - p_tgt))
            self.get_logger().info(f"[{i+1}/{len(targets_raw)}] Done in {t_el:.2f}s. Final tip→dummy dist = {pos_res*1000:.1f} mm")

        self.get_logger().info("All targets processed.")

    def _to_parent_frame(self, pos_in_frame: List[float], rpy_in_frame: Tuple[float, float, float], frame: str):
        parent = self.sim.target_parent
        sim = self.sim.sim

        if frame in ("scene", "world", "global"):
            ref = -1
        elif frame in ("ur3", "base_link", "ur", "base"):
            ref = self.sim.base
        else:
            ref = -1

        def T_a_from_b(a_handle, b_handle):
            p_ba = sim.getObjectPosition(a_handle, b_handle)
            r_ba = sim.getObjectOrientation(a_handle, b_handle)
            Tb_a = np.eye(4)
            Tb_a[:3, :3] = tft.euler_matrix(*r_ba)[:3, :3]
            Tb_a[:3, 3]  = np.array(p_ba, dtype=float)
            return np.linalg.inv(Tb_a)  # ^a T_b

        if ref == parent:
            return np.array(pos_in_frame, float), tuple(rpy_in_frame)

        T_parent_frame = T_a_from_b(parent, ref)
        Rf = tft.euler_matrix(*rpy_in_frame)[:3, :3]
        Tf = np.eye(4); Tf[:3, :3] = Rf; Tf[:3, 3] = np.array(pos_in_frame, float)
        Tp = T_parent_frame @ Tf
        pos_parent = Tp[:3, 3]
        rpy_parent = tuple(tft.euler_from_matrix(Tp))
        return pos_parent, rpy_parent


# -------- main --------
def main():
    rclpy.init()
    node = VisionTargetsIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
