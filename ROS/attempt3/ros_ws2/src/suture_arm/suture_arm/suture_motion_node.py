#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
suture_motion_node.py — ROS 2 node that moves UR3 to published suture poses.

Topic (subscribe):
  - /suture_arm/suture_path : geometry_msgs/PoseArray   (world-frame poses)

Scene assumptions (CoppeliaSim):
  - A dummy named '/UR3_target' is the IK target of the UR3 chain.
  - Simulation is running (node will start it if needed).
  - Poses published are already in WORLD coordinates (same as sim.getObjectPosition(..., -1)).

Parameters (ROS 2, can set via launch or `ros2 param set`):
  - steps_per_move (int, default 8): sim steps between setpoints
  - dwell_sec      (float, default 0.05): pause at contact poses (every 3rd point if following approach/contact/retract triplets)
  - target_name    (str, default '/UR3_target'): IK target path
  - sim_host       (str, default '127.0.0.1')
  - sim_port       (int, default 23000)
"""
import time
from typing import List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import numpy as np

# CoppeliaSim ZMQ remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


class SutureMotionNode(Node):
    def __init__(self):
        super().__init__('suture_motion_node')

        # --- Parameters ---
        self.declare_parameter('steps_per_move', 8)
        self.declare_parameter('dwell_sec', 0.05)
        self.declare_parameter('target_name', '/UR3_target')
        self.declare_parameter('sim_host', '127.0.0.1')
        self.declare_parameter('sim_port', 23000)

        self.steps_per_move: int = int(self.get_parameter('steps_per_move').value)
        self.dwell_sec: float = float(self.get_parameter('dwell_sec').value)
        self.target_name: str = str(self.get_parameter('target_name').value)
        self.sim_host: str = str(self.get_parameter('sim_host').value)
        self.sim_port: int = int(self.get_parameter('sim_port').value)

        # --- Connect to CoppeliaSim ---
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError:
            from zmqRemoteApi import RemoteAPIClient  # legacy
        self.client = RemoteAPIClient(self.sim_host, self.sim_port)
        self.sim = self.client.require("sim")

        # Ensure sim running & stepping
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception as e:
            self.get_logger().warn(f"Could not ensure simulation running: {e}")
        try:
            if hasattr(self.sim, "setStepping"):
                self.sim.setStepping(True)
        except Exception as e:
            self.get_logger().warn(f"Could not set stepping: {e}")

        # --- Resolve UR3 target handle robustly ---
        self.ik_target = None

        # 1) Try the provided name and common variants
        guess_list = [
            self.target_name, self.target_name.lstrip('/'),
            "/UR3_target", "UR3_target", "/UR3_target#0",
            "/tipTarget", "tipTarget", "/ee_target", "ee_target",
            "/needle_target", "needle_target"
        ]
        for cand in guess_list:
            try:
                self.ik_target = self.sim.getObject(cand)
                self.get_logger().info(f"IK target resolved as '{cand}'")
                break
            except Exception:
                pass

        # 2) If not found, scan all dummies and pick the best match
        if self.ik_target is None:
            try:
                # sim.object_dummy_type == 15 in CoppeliaSim (avoid hard-coding if constant missing)
                obj_type = getattr(self.sim, "object_dummy_type", 15)
                all_dummies = self.sim.getObjects(obj_type)
                best = None; best_score = -1
                for h in all_dummies:
                    name = self.sim.getObjectAlias(h)
                    nlow = name.lower()
                    score = 0
                    if "target" in nlow: score += 2
                    if "ur3" in nlow:    score += 2
                    if "ee" in nlow or "tip" in nlow or "needle" in nlow: score += 1
                    if score > best_score:
                        best = h; best_score = score
                if best is not None:
                    self.ik_target = best
                    nm = self.sim.getObjectAlias(best)
                    self.get_logger().warn(f"IK target auto-selected: '{nm}' (score={best_score})")
            except Exception as e:
                self.get_logger().warn(f"Auto-scan for IK target failed: {e}")

        if self.ik_target is None:
            raise RuntimeError(f"UR3 IK target not found. "
                            f"Set 'target_name' param to your dummy path (e.g. '/tipTarget').")

        # Cache current orientation to reuse at waypoints
        try:
            self.cur_quat = self.sim.getObjectQuaternion(self.ik_target, -1)
        except Exception:
            self.cur_quat = [0.0, 0.0, 0.0, 1.0]

        # Subscriber
        from geometry_msgs.msg import PoseArray
        self.sub = self.create_subscription(PoseArray, '/suture_arm/suture_path',
                                            self._on_path, 1)
        self.get_logger().info("suture_motion_node ready. Subscribed to /suture_arm/suture_path.")

    # ---- utilities ----
    def _step(self):
        try:
            if hasattr(self.sim, "step"):
                self.sim.step()
            else:
                time.sleep(0.01)
        except Exception:
            time.sleep(0.01)

    def _move_through(self, points_w: List[List[float]]):
        spm = max(1, int(self.steps_per_move))
        dwell = max(0.0, float(self.dwell_sec))
        for i, p in enumerate(points_w):
            # position is [x,y,z] meters in world frame
            self.sim.setObjectPosition(self.ik_target, -1, p)
            self.sim.setObjectQuaternion(self.ik_target, -1, self.cur_quat)
            for _ in range(spm):
                self._step()
            # Dwell at likely "contact" poses (every 2nd in triples: approach/contact/retract)
            if dwell > 0.0 and (i % 3) == 1:
                t_end = time.time() + dwell
                while time.time() < t_end:
                    self._step()

    # ---- callback ----
    def _on_path(self, msg: PoseArray):
        n = len(msg.poses)
        if n == 0:
            self.get_logger().warn("Received empty PoseArray; ignoring.")
            return

        # Convert to list of [x,y,z] in meters (world frame)
        points = []
        for i, pose in enumerate(msg.poses):
            # Use published orientation if provided; otherwise keep cur_quat
            pos = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
            # (optional) If you want to follow orientation too, uncomment:
            # q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            points.append(pos)

        self.get_logger().info(f"Executing suture path with {n} waypoints...")
        try:
            self._move_through(points)
            self.get_logger().info("Suture path execution finished.")
        except Exception as e:
            self.get_logger().error(f"Motion error: {e}")


def main():
    rclpy.init()
    try:
        node = SutureMotionNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"[suture_motion_node] fatal: {e}", flush=True)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
