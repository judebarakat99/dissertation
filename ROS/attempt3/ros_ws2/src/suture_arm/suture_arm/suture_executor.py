#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Suture path executor (Option A: separate node)

- Subscribes:  /suture_cuts (std_msgs/String, JSON from vision_web.py)
- Publishes:   /suture_waypoints (geometry_msgs/PoseArray) for viz/controller
- Optional:    /suture_stop (std_msgs/Bool) to stop mid-execution

CoppeliaSim objects (defaults set for your scene):
  TIP_NAME       = "UR3_tip"                 # IK tip dummy
  TARGET_NAME    = "dummy needle target"     # IK target dummy (this script moves it)
  MAT_FRAME_NAME = "mat"                     # parent of suture_pad[1]
  BASE_FRAME_NAME= ""                        # if empty, auto-detect by walking parents of TIP

Motion tunables (env):
  APPROACH_M=0.010          # hover height above pad (meters)
  DEPTH_M=0.003             # peck depth into pad (meters)
  DWELL_S=0.15              # dwell time at bottom (seconds)
  TRAVEL_STEP=0.005         # interpolation step (meters)
  DT=0.03                   # sleep per sub-step (seconds)
  ORIENT_TOWARDS_TANGENT=1  # 1: align X with path tangent; 0: fixed orientation
  DRY_RUN=0                 # 1: don't move in sim; just publish PoseArray
  CSIM_HOST=127.0.0.1
  CSIM_PORT=23000
"""

import os, json, math, time
from typing import List, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as MsgString, Bool as MsgBool
from geometry_msgs.msg import Pose, PoseArray

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


# ------------------------- math helpers -------------------------

def _R_from_quat(x, y, z, w) -> np.ndarray:
    x, y, z, w = map(float, (x, y, z, w))
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ], dtype=float)

def _quat_from_R(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.array(R, dtype=float).reshape(3,3)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return (float(x), float(y), float(z), float(w))

def _tangent_R(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray, z_axis=np.array([0.,0.,1.])):
    t = (p_next - p_prev)
    n = np.linalg.norm(t)
    t = np.array([1.,0.,0.]) if n < 1e-9 else t / n
    z = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    x = t - np.dot(t, z)*z
    n = np.linalg.norm(x)
    if n < 1e-9:
        x = np.array([1.,0.,0.])
        if abs(np.dot(x,z)) > 0.9: x = np.array([0.,1.,0.])
        x = x - np.dot(x,z)*z
        x /= (np.linalg.norm(x) + 1e-12)
    else:
        x /= n
    y = np.cross(z, x)
    return np.column_stack([x, y, z])

def _slerp(q0, q1, u):
    q0 = np.array(q0, dtype=float); q1 = np.array(q1, dtype=float)
    dot = np.dot(q0, q1)
    if dot < 0.0: q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + u*(q1 - q0)
        return (q / np.linalg.norm(q)).tolist()
    th0 = math.acos(dot); s0 = math.sin(th0)
    q = (math.sin((1-u)*th0)/s0)*q0 + (math.sin(u*th0)/s0)*q1
    return (q / np.linalg.norm(q)).tolist()

def _interp_pose(p0, q0, p1, q1, step_m=0.005):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    q0 = np.array(q0, float); q1 = np.array(q1, float)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(1, int(math.ceil(dist / max(1e-5, step_m))))
    for i in range(n+1):
        u = i / n
        p = (1-u)*p0 + u*p1
        q = _slerp(q0, q1, u)
        yield (p, tuple(q))

def _pose_to_list(pos, quat):
    x,y,z = [float(t) for t in pos]
    qx,qy,qz,qw = [float(t) for t in quat]
    return [x,y,z,qx,qy,qz,qw]


# ------------------------- executor node -------------------------

class SutureExecutor(Node):
    def __init__(self):
        super().__init__("suture_executor")

        # Scene names (yours)
        self.tip_name     = os.getenv("TIP_NAME", "UR3_tip")
        self.target_name  = os.getenv("TARGET_NAME", "dummy needle target")
        self.mat_name     = os.getenv("MAT_FRAME_NAME", "mat")
        self.base_name    = os.getenv("BASE_FRAME_NAME", "")  # empty => auto-detect

        # Motion params
        self.approach_m   = float(os.getenv("APPROACH_M", "0.010"))
        self.depth_m      = float(os.getenv("DEPTH_M",    "0.003"))
        self.dwell_s      = float(os.getenv("DWELL_S",    "0.15"))
        self.travel_step  = float(os.getenv("TRAVEL_STEP","0.005"))
        self.dt           = float(os.getenv("DT",         "0.03"))
        self.align_tangent= bool(int(os.getenv("ORIENT_TOWARDS_TANGENT", "1")))
        self.dry_run      = bool(int(os.getenv("DRY_RUN","0")))

        # ROS I/O
        self.sub_cuts = self.create_subscription(MsgString, "/suture_cuts", self.on_cuts, 10)
        self.sub_stop = self.create_subscription(MsgBool,   "/suture_stop", self.on_stop, 10)
        self.pub_waypoints = self.create_publisher(PoseArray, "/suture_waypoints", 10)

        # Sim handles
        self.sim_client = None
        self.sim = None
        self.h_base = None
        self.h_mat  = None
        self.h_tip  = None
        self.h_target = None

        self._stop_flag = False
        self.get_logger().info("SutureExecutor ready (Option A)")

    # ---- sim helpers ----
    def ensure_sim(self):
        if self.sim is not None:
            return
        host = os.getenv("CSIM_HOST", "127.0.0.1")
        port = int(os.getenv("CSIM_PORT", "23000"))
        self.sim_client = RemoteAPIClient(host, port)
        self.sim = self.sim_client.require("sim")

        def _get(nm):
            try: return self.sim.getObject(nm)
            except Exception:
                try: return self.sim.getObject(nm.lstrip("/"))
                except Exception: return None

        self.h_tip    = _get(self.tip_name)
        self.h_target = _get(self.target_name)
        self.h_mat    = _get(self.mat_name)
        if self.h_tip is None or self.h_target is None or self.h_mat is None:
            raise RuntimeError(f"Missing handles: tip={self.h_tip}, target={self.h_target}, mat={self.h_mat}")

        if self.base_name:
            self.h_base = _get(self.base_name)
            if self.h_base is None:
                raise RuntimeError(f"BASE_FRAME_NAME '{self.base_name}' not found")
        else:
            # auto-detect base by walking parents from tip to root
            h = self.h_tip
            parent = self.sim.getObjectParent(h)
            while parent != -1:
                h = parent
                parent = self.sim.getObjectParent(h)
            self.h_base = h
            self.get_logger().info(f"Auto-detected base handle: {self.h_base}")

        # ensure sim is running
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

    def get_pose_rel(self, obj, ref):
        return self.sim.getObjectPose(obj, ref)  # [x,y,z,qx,qy,qz,qw]

    def set_pose_rel(self, obj, ref, pose_xyzw):
        self.sim.setObjectPose(obj, ref, pose_xyzw)

    # ---- ROS utils ----
    def on_stop(self, msg: MsgBool):
        if msg.data:
            self._stop_flag = True
            self.get_logger().warn("Emergency stop requested.")

    def publish_waypoints(self, poses_base: List[List[float]]):
        arr = PoseArray()
        arr.header.frame_id = "base"
        for p in poses_base:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p[0], p[1], p[2]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = p[3], p[4], p[5], p[6]
            arr.poses.append(pose)
        self.pub_waypoints.publish(arr)

    # ---- main callback ----
    def on_cuts(self, msg: MsgString):
        self.ensure_sim()
        self._stop_flag = False

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Bad JSON: {e}")
            return

        cuts = data.get("cuts", [])
        if not cuts or not cuts[0].get("polyline"):
            self.get_logger().warn("No polyline")
            return

        poly = np.array(cuts[0]["polyline"], dtype=float)  # Nx2 in MAT frame (meters)
        if poly.shape[0] < 2:
            self.get_logger().warn("Polyline too short")
            return

        # orientations in MAT frame
        mat_z = np.array([0.,0.,1.])
        R_list = []
        for i in range(len(poly)):
            p_prev = poly[max(i-1,0)]
            p_curr = poly[i]
            p_next = poly[min(i+1, len(poly)-1)]
            if self.align_tangent:
                R = _tangent_R(
                    np.array([p_prev[0], p_prev[1], 0.0]),
                    np.array([p_curr[0], p_curr[1], 0.0]),
                    np.array([p_next[0], p_next[1], 0.0]),
                    z_axis=mat_z
                )
            else:
                R = np.eye(3)
            R_list.append(R)

        approach_z = +abs(self.approach_m)
        peck_z     = -abs(self.depth_m)

        # tagged sequence in MAT frame
        seq_mat: List[Tuple[List[float], str]] = []
        def push(x,y,z,R, tag):
            q = _quat_from_R(R)
            seq_mat.append(([x,y,z,q[0],q[1],q[2],q[3]], tag))

        # start hover above first point
        x0,y0 = poly[0]
        push(x0, y0, approach_z, R_list[0], "start_hover")

        for i in range(len(poly)):
            x,y = poly[i]; R = R_list[i]
            push(x, y, approach_z, R, "travel")
            push(x, y, peck_z,     R, "down")
            push(x, y, approach_z, R, "up")

        # MAT -> BASE transform
        base_T_mat = self.get_pose_rel(self.h_mat, self.h_base)
        p = np.array(base_T_mat[:3], float)
        q = base_T_mat[3:7]
        Rb = _R_from_quat(*q)
        T_base_mat = np.eye(4,float); T_base_mat[:3,:3] = Rb; T_base_mat[:3,3] = p

        poses_base: List[List[float]] = []
        tags: List[str] = []
        for (pxyzw, tag) in seq_mat:
            px,py,pz,qx,qy,qz,qw = pxyzw
            T_mat_tcp = np.eye(4,float)
            T_mat_tcp[:3,:3] = _R_from_quat(qx,qy,qz,qw)
            T_mat_tcp[:3,3]  = np.array([px,py,pz],float)
            T_base_tcp = T_base_mat @ T_mat_tcp
            pos = T_base_tcp[:3,3]
            quat = _quat_from_R(T_base_tcp[:3,:3])
            poses_base.append([pos[0],pos[1],pos[2], *quat])
            tags.append(tag)

        # publish for visualization / external controllers
        self.publish_waypoints(poses_base)
        if self.dry_run:
            self.get_logger().info(f"[DRY] Generated {len(poses_base)} waypoints (peck profile).")
            return

        # execute with interpolation and dwell at each 'down'
        self.get_logger().info(f"Executing {len(poses_base)} waypoints...")
        last_pose = self.get_pose_rel(self.h_target, self.h_base)
        curr_p = np.array(last_pose[:3], float)
        curr_q = tuple(last_pose[3:7])

        for i, pb in enumerate(poses_base):
            if self._stop_flag:
                self.get_logger().warn("Stopped by /suture_stop")
                break

            next_p = np.array(pb[:3], float)
            next_q = tuple(pb[3:7])

            for pos, quat in _interp_pose(curr_p, curr_q, next_p, next_q, step_m=self.travel_step):
                if self._stop_flag:
                    break
                self.set_pose_rel(self.h_target, self.h_base, _pose_to_list(pos, quat))
                try:
                    if hasattr(self.sim, "step"): self.sim.step()
                except Exception:
                    pass
                time.sleep(self.dt)

            if tags[i] == "down" and not self._stop_flag:
                time.sleep(self.dwell_s)

            curr_p, curr_q = next_p, next_q

        self.get_logger().info("Execution complete.")


# ------------------------- entry point -------------------------

def main():
    rclpy.init()
    node = SutureExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
