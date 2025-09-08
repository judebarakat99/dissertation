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
#
# Positional error tolerance for the inverse kinematics solver.  When the
# positional error (tip→target distance) falls below this threshold, the
# solver will move on to the next target.  The value of 0.004 m
# corresponds to 4 mm, relaxing the previous ~1.8 mm tolerance.
IK_POS_THR_M          = 0.004    # 0.004 m = 4 mm
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
    def __init__(self, node: Node, tip_name: str = "needle_tip"):
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

        # --- TIP selection: prefer needle_tip, fallback to UR3_tip ---
        tip_candidates = [
            f"{ROBOT_BASE_PATH}/{tip_name}",
            f"/{tip_name}",
            tip_name,
            f"{tip_name}#0",
        ]
        self.tip = self._resolve_one(tip_candidates)
        used_tip_name = tip_name

        if self.tip is None:
            # fallback list for UR3_tip (legacy)
            fallback = [f"{ROBOT_BASE_PATH}/UR3_tip", "/UR3_tip", "UR3_tip", "UR3_tip#0"]
            self.tip = self._resolve_one(fallback)
            used_tip_name = "UR3_tip"

        if self.tip is None:
            raise RuntimeError(f"Neither '{tip_name}' nor 'UR3_tip' found. Check the scene object names.")

        node.get_logger().info(f"Using IK tip: {used_tip_name}")

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

        ik.setGroupCalculation(self.ikEnv, self.ikGroup, ik.method_damped_least_squares, IK_DLS_DAMPING, 50)

        # Constraints: position + alphaBeta (tilt). Intentionally ignore gamma (yaw).
        self.CONSTR_POS_TILT = ik.constraint_x | ik.constraint_y | ik.constraint_z | ik.constraint_alpha_beta

        # Map the current scene chain into IK (base→tip) with target
        self.ikElement = ik.addElementFromScene(self.ikEnv, self.ikGroup, self.base, self.tip, self.target, self.CONSTR_POS_TILT)

        # Motorized pos-control for all joints
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
                sim.setJointTargetForce(j, 120.0)
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

    def ik_servo_to_target(self, timeout_s: float = 240.0) -> Tuple[int, float, float]:
        """
        Run the simIK solver until the tip is sufficiently close to the target or
        a timeout/hard iteration cap is reached.

        Args:
            timeout_s: maximum wall-clock seconds to run the servo.

        Returns:
            A tuple ``(steps, pos_err, tilt_err)`` where ``steps`` is the number of
            simulation ticks taken, ``pos_err`` is the final positional error in
            metres, and ``tilt_err`` is the final tilt error in radians.  The
            positional error tolerance is determined by ``IK_POS_THR_M``.
        """
        ik = self.simIK
        sim = self.sim

        t0 = time.time()
        steps = 0
        opts = {"syncWorlds": True}  # pulls scene→IK, solves, pushes IK→scene

        # initialise errors in case the loop exits immediately
        pos_err = float('inf')
        tilt_err = float('inf')
        while steps < IK_TICK_COUNT:
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
            # tilt error defined as the angle between z-axis vectors (alpha/beta)
            dot_val = float(np.clip(np.dot(z_tip, z_tgt), -1.0, 1.0))
            tilt_err = math.acos(dot_val)

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

        # Ensure the simulation has settled by stepping a few more times
        for _ in range(3):
            sim.step()

        # final error computation in case loop broke early
        p_tip, R_tip = self.get_tip_pose_in_base()
        p_tgt, R_tgt = self.get_target_pose_in_base()
        pos_err = float(np.linalg.norm(p_tgt - p_tip))
        z_tip = R_tip[:, 2] / max(1e-9, np.linalg.norm(R_tip[:, 2]))
        z_tgt = R_tgt[:, 2] / max(1e-9, np.linalg.norm(R_tgt[:, 2]))
        dot_val = float(np.clip(np.dot(z_tip, z_tgt), -1.0, 1.0))
        tilt_err = math.acos(dot_val)

        return steps, pos_err, tilt_err

    def home_all_joints_zero(self):
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
        self.declare_parameter("tip_name", "needle_tip")

        self.work_z          = float(self.get_parameter("work_z").value)
        self.down_rpy        = tuple(float(x) for x in self.get_parameter("down_rpy").value)
        self.force_down      = bool(self.get_parameter("force_down_orientation").value)
        self.force_work_z    = bool(self.get_parameter("force_work_z").value)
        self.home_on_start   = bool(self.get_parameter("home_on_start").value)
        damping_in           = float(self.get_parameter("ik_damping").value)
        tip_name             = str(self.get_parameter("tip_name").value)

        self.get_logger().info(f"Connecting to CoppeliaSim at {CSIM_HOST}:{CSIM_PORT} ...")
        self.sim = CoppeliaIK(self, tip_name=tip_name)

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

        # Handle normalized/unit-frame targets.  When 'frame' is 'unit' or 'normalized',
        # the payload's positions are normalized in [0,1] for both x and y axes.  We
        # convert these to world coordinates using x→[-1,0] and y→[-0.5,0.5], and fix
        # z to 0.02 m.  The mapping is: x_world = -1 + x_norm, y_world = -0.5 + y_norm.
        # We then transform into the dummy's parent frame and servo to each point.
        if frame in ("unit", "normalized", "unit_square", "unitframe", "unit-frame"):
            # When the payload frame is 'unit' or similar, treat each position's x and y
            # component as normalized coordinates in [0,1], with (0,0) at the top-left
            # corner of the vision sensor view and (1,1) at the bottom-right.  The
            # mapping to world coordinates is:
            #   x_world = -x_norm           # 0→0, 1→-1 (sensor x axis left-to-right → world x axis right-to-left)
            #   y_world =  0.5 - y_norm     # 0→0.5, 1→-0.5 (sensor y axis top-to-bottom → world y axis top positive)
            # We then convert that world-frame pose into the target dummy's parent
            # frame and set its Z to a fixed relative offset (-0.02303) so that its
            # world Z ends up at approximately +0.02 m (given the base height is
            # around +0.04303 m).  If XY are out of reach, we project them into
            # the allowable workspace while leaving Z untouched.
            # Constant Z relative to the parent (UR3 base).  If the base is roughly
            # at Z=+0.04303 in the scene, then world_z=+0.02 corresponds to
            # parent_z=-0.02303.
            z_parent_const = -0.02303
            for i, trg in enumerate(targets_raw):
                # Extract normalized coordinates directly; avoid to_pose_dict since
                # to_pose_dict expects length ≥3 for pos.
                pos_norm = None
                if isinstance(trg, dict):
                    pos_norm = trg.get("pos")
                elif isinstance(trg, (list, tuple)):
                    pos_norm = trg
                if not isinstance(pos_norm, (list, tuple)) or len(pos_norm) < 2:
                    self.get_logger().warn(f"Skipping malformed unit target #{i}: pos missing or too short")
                    continue
                try:
                    x_norm = float(pos_norm[0])
                    y_norm = float(pos_norm[1])
                except Exception as e:
                    self.get_logger().warn(f"Skipping malformed unit target #{i}: {e}")
                    continue
                # Clamp normalized values into [0,1].
                if x_norm < 0.0: x_norm = 0.0
                elif x_norm > 1.0: x_norm = 1.0
                if y_norm < 0.0: y_norm = 0.0
                elif y_norm > 1.0: y_norm = 1.0
                # Map to world coordinates.  
                x_world = -x_norm 
                y_world = y_norm
                pos_world = [x_world, y_world, 0.02]  # constant world Z
                # Determine the desired orientation.  If force_down is True, use
                # the configured down_rpy; otherwise, attempt to use an RPY from
                # the message or keep the current orientation of the dummy.
                if self.force_down:
                    rpy_in = self.down_rpy
                    want_orientation = True
                else:
                    rpy_raw = None
                    if isinstance(trg, dict):
                        rpy_raw = trg.get("rpy")
                    want_orientation = rpy_raw is not None
                    if want_orientation:
                        try:
                            rpy_in = tuple(float(v) for v in rpy_raw)
                        except Exception:
                            rpy_in = self.sim.sim.getObjectOrientation(self.sim.target, self.sim.target_parent)
                            want_orientation = False
                    else:
                        rpy_in = self.sim.sim.getObjectOrientation(self.sim.target, self.sim.target_parent)
                # Convert the world pose into the dummy's parent frame.  We use
                # 'scene' for the frame argument, since pos_world is in the scene/world frame.
                pos_parent, rpy_parent = self._to_parent_frame(pos_world, rpy_in, "scene")
                # Override the Z component to a fixed relative offset so that the
                # world Z ends up at +0.02.  We rely on the fact that this node
                # uses UR3_target as a child of the UR3 base.
                pos_parent = [pos_parent[0], pos_parent[1], z_parent_const]
                # We no longer enforce a radial limit on unit-frame targets.  The dummy
                # will be placed exactly at the mapped coordinates even if they are
                # outside the robot's reachable workspace.  This may cause the IK
                # solver to fail to converge if the point is unreachable, but it
                # preserves the intended mapping for debugging/calibration.
                # Log and execute the servo.
                self.get_logger().info(
                    f"[unit {i+1}/{len(targets_raw)}] Move dummy → pos={np.round(pos_parent,5).tolist()}, "
                    f"rpy={tuple(round(v,5) for v in rpy_parent)} (normalized frame)"
                )
                self.sim.set_dummy_pose(pos_parent, rpy_parent if want_orientation else None)
                self.get_logger().info(f"[unit {i+1}/{len(targets_raw)}] simIK servo to dummy ...")
                t0 = time.time()
                # run IK and capture final step count and errors
                steps, pos_err, tilt_err = self.sim.ik_servo_to_target(timeout_s=240.0)
                t_el = time.time() - t0
                within_threshold = (pos_err <= IK_POS_THR_M)
                self.get_logger().info(
                    f"[unit {i+1}/{len(targets_raw)}] Done in {t_el:.2f}s. pos_err={pos_err*1000:.1f} mm, "
                    f"tilt_err={math.degrees(tilt_err):.1f} deg, ticks={steps}, "
                    f"within_threshold={within_threshold}"
                )
            self.get_logger().info("All unit targets processed.")
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

            # If the dummy is parented to the UR3 base, we only adjust the Z
            # coordinate to match the configured work plane (if force_work_z is
            # enabled).  We no longer project XY into a safe radius.  This means
            # that unreachable targets will be attempted as-is, which may
            # result in the IK solver failing to converge or reaching a joint limit.
            if self.sim.target_parent == self.sim.base:
                pos_parent = [pos_parent[0], pos_parent[1], self.work_z]

            self.get_logger().info(
                f"[{i+1}/{len(targets_raw)}] Move dummy → pos={np.round(pos_parent,5).tolist()}, "
                f"rpy={tuple(round(v,5) for v in rpy_parent)} (frame={frame}, tilt-only={want_orientation})"
            )
            self.sim.set_dummy_pose(pos_parent, rpy_parent if want_orientation else None)

            # Run simIK servo and collect final metrics
            self.get_logger().info(f"[{i+1}/{len(targets_raw)}] simIK servo to dummy ...")
            t0 = time.time()
            steps, pos_err, tilt_err = self.sim.ik_servo_to_target(timeout_s=240.0)
            t_el = time.time() - t0
            within_threshold = (pos_err <= IK_POS_THR_M)
            self.get_logger().info(
                f"[{i+1}/{len(targets_raw)}] Done in {t_el:.2f}s. pos_err={pos_err*1000:.1f} mm, "
                f"tilt_err={math.degrees(tilt_err):.1f} deg, ticks={steps}, "
                f"within_threshold={within_threshold}"
            )

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