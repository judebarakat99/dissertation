#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Suture path executor with simIK (no args needed)

Usage:
  ros2 run suture_arm suture_executor

We set the target pose and call simIK.handleGroup() so the **robot joints**
move (not the target dummy). Auto-finds UR3_tip, dummy needle target, and mat.
"""

import json
import math
import time
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from std_msgs.msg import Bool as MsgBool, String as MsgString

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


# ---------------- math helpers ----------------
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

def _interp_pose(p0, q0, p1, q1, step_m=0.02):  # faster default
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


# ---------------- executor node ----------------
class SutureExecutor(Node):
    def __init__(self):
        super().__init__("suture_executor")

        # Motion params (fast but smooth)
        self.approach_m   = 0.010
        self.depth_m      = 0.003
        self.dwell_s      = 0.05   # shorter dwell
        self.travel_step  = 0.02   # 2 cm per sub-step
        self.dt           = 0.0    # no extra sleep
        self.align_tangent= True
        self.dry_run      = False

        # ROS I/O
        self.sub_cuts = self.create_subscription(MsgString, "/suture_cuts", self.on_cuts, 10)
        self.sub_stop = self.create_subscription(MsgBool,   "/suture_stop", self.on_stop, 10)
        self.pub_waypoints = self.create_publisher(PoseArray, "/suture_waypoints", 10)

        # Sim & IK
        self.sim_client = None
        for_ik = None
        self.sim = None
        self.simIK = None
        self.ik_env = None
        self.ik_group = None

        self.h_base = None
        self.h_mat  = None
        self.h_tip  = None
        self.h_target = None

        # What frame the IK element uses as "base" (-1 means world)
        self.ik_base_ref = -1

        self._stop_flag = False
        self.get_logger().info("SutureExecutor with simIK ready (no env args)")

    # -------- sim helpers --------
    def ensure_sim(self):
        if self.sim is not None:
            return
        host = "127.0.0.1"
        port = 23000
        self.sim_client = RemoteAPIClient(host, port)
        self.sim = self.sim_client.require("sim")

        # simIK plugin
        try:
            self.simIK = self.sim_client.require("simIK")
        except Exception as e:
            raise RuntimeError(f"simIK plugin not available: {e}")

        # make sure sim is running; free-run for speed
        try:
            if hasattr(self.sim, "setStepping"):
                self.sim.setStepping(False)
        except Exception:
            pass
        try:
            st = self.sim.getSimulationState()
            if st in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception:
            pass

        # 1) TIP
        self.h_tip = self._resolve_exact_or_variants("UR3_tip")
        if self.h_tip is None:
            raise RuntimeError("Could not find 'UR3_tip'.")

        # 2) MAT
        self.h_mat = self._resolve_exact_or_variants("mat")
        if self.h_mat is None:
            self.h_mat = self._search_near(self.h_tip, [], ["mat", "pad", "suture_pad"])
        if self.h_mat is None:
            raise RuntimeError("Could not find 'mat' dummy.")

        # 3) TARGET
        self.h_target = self._resolve_exact_or_variants("dummy needle target")
        if self.h_target is None:
            self.h_target = self._search_descendant_dummy(self.h_tip, ["needle","target"], [])
            if self.h_target is None:
                self.h_target = self._search_descendant_dummy(self.h_tip, [], ["needle","target","biopsy","forcep"])
        if self.h_target is None:
            self._dump_descendants(self.h_tip, "tip subtree (looking for '*needle*' AND/OR '*target*')")
            raise RuntimeError("Could not find the target dummy under the tool.")

        # 4) BASE (walk to root for fallback signature)
        h = self.h_tip
        parent = self.sim.getObjectParent(h)
        while parent != -1:
            h = parent
            parent = self.sim.getObjectParent(h)
        self.h_base = h  # model root, used only if world-base fails

        # 5) Build IK task: TIP follows TARGET (full pose constraint)
        self.ik_env = self.simIK.createEnvironment()
        self.ik_group = self.simIK.createGroup(self.ik_env)
        self.simIK.setGroupCalculation(self.ik_env, self.ik_group,
                                       self.simIK.method_damped_least_squares, 0.01, 10)

        # ---- prefer world (-1) as IK base
        try:
            _el = self.simIK.addIkElementFromScene(
                self.ik_env, self.ik_group, self.h_tip, -1, self.h_target, self.simIK.constraint_pose
            )
            self.ik_base_ref = -1
        except Exception as e_world:
            # Fallback: use discovered model root as base
            self.get_logger().warn(f"World-base IK element failed ({e_world}); "
                                   f"trying model-root base (handle {self.h_base}).")
            _el = self.simIK.addIkElementFromScene(
                self.ik_env, self.ik_group, self.h_tip, self.h_base, self.h_target, self.simIK.constraint_pose
            )
            self.ik_base_ref = self.h_base

        if _el is None or int(_el) < 0:
            raise RuntimeError("simIK.addIkElementFromScene failed: got invalid element handle.")

        # Slightly higher weight on orientation
        self.simIK.setElementWeights(self.ik_env, self.ik_group, _el, [1,1,1, 2,2,2])

        base_label = "world(-1)" if self.ik_base_ref == -1 else f"handle {self.ik_base_ref}"
        self.get_logger().info(
            f"Handles: tip={self.h_tip}, target={self.h_target}, mat={self.h_mat}, ik_base={base_label}"
        )
        self.get_logger().info("simIK environment initialized.")

    def _resolve_exact_or_variants(self, name: str) -> Optional[int]:
        tried = set()
        for v in (name, name.lstrip("/"), f"{name}#0", f"/{name}", f"/{name}#0"):
            if v in tried: continue
            tried.add(v)
            try:
                return self.sim.getObject(v)
            except Exception:
                pass
        return None

    def _search_descendant_dummy(self, root: int, must_have_all: List[str], may_have_any: List[str]) -> Optional[int]:
        try:
            dummy_type = getattr(self.sim, "object_dummy_type", 5)
            hs = self.sim.getObjectsInTree(root, dummy_type, 0)
        except Exception:
            hs = [root]
            q = [root]
            for _ in range(3):
                nq = []
                for h in q:
                    try:
                        ch = self.sim.getObjectsInTree(h, 0, 2)
                        for c in ch:
                            if c not in hs:
                                hs.append(c); nq.append(c)
                    except Exception:
                        pass
                q = nq

        def is_dummy(h) -> bool:
            try:
                return self.sim.getObjectType(h) == getattr(self.sim, "object_dummy_type", 5)
            except Exception:
                return True

        cands = []
        for h in hs:
            if not is_dummy(h): continue
            try:
                alias = self.sim.getObjectAlias(h, 1)
            except Exception:
                try: alias = self.sim.getObjectAlias(h)
                except Exception: continue
            al = alias.lower()
            ok_all = all(tok in al for tok in must_have_all) if must_have_all else True
            ok_any = any(tok in al for tok in may_have_any) if may_have_any else True
            if ok_all and ok_any:
                score = (len(must_have_all) + sum(tok in al for tok in may_have_any), -len(alias))
                cands.append((score, h))
        if not cands: return None
        cands.sort(reverse=True)
        return cands[0][1]

    def _search_near(self, root: int, must_have_all: List[str], may_have_any: List[str]) -> Optional[int]:
        try:
            hs = self.sim.getObjectsInTree(root, 0, 0)
        except Exception:
            hs = [root]
            q = [root]
            for _ in range(3):
                nq = []
                for h in q:
                    try:
                        ch = self.sim.getObjectsInTree(h, 0, 2)
                        for c in ch:
                            if c not in hs:
                                hs.append(c); nq.append(c)
                    except Exception:
                        pass
                q = nq

        for h in hs:
            try:
                alias = self.sim.getObjectAlias(h, 1)
            except Exception:
                try: alias = self.sim.getObjectAlias(h)
                except Exception: continue
            al = alias.lower()
            ok_all = all(tok in al for tok in must_have_all) if must_have_all else True
            ok_any = any(tok in al for tok in may_have_any) if may_have_any else True
            if ok_all and ok_any:
                return h
        return None

    def _dump_descendants(self, root: Optional[int], label="subtree"):
        if root is None:
            self.get_logger().error(f"No {label} to dump.")
            return
        self.get_logger().error(f"Listing descendant dummies in {label}:")
        try:
            dummy_type = getattr(self.sim, "object_dummy_type", 5)
            hs = self.sim.getObjectsInTree(root, dummy_type, 0)
            for i, h in enumerate(hs[:80]):
                try:
                    a = self.sim.getObjectAlias(h, 1)
                except Exception:
                    a = self.sim.getObjectAlias(h)
                self.get_logger().info(f"  [{i:03d}] {a}")
        except Exception as e:
            self.get_logger().error(f"Descendant dump failed: {e}")

    def get_pose_rel(self, obj, ref):
        return self.sim.getObjectPose(obj, ref)  # [x,y,z,qx,qy,qz,qw]

    def set_target_pose_and_solve(self, pose_xyzw):
        """Set target pose (relative to IK base) and run IK once."""
        self.sim.setObjectPose(self.h_target, self.ik_base_ref, pose_xyzw)
        self.simIK.handleGroup(self.ik_env, self.ik_group)

    # -------- ROS utils --------
    def on_stop(self, msg: MsgBool):
        if msg.data:
            self._stop_flag = True
            self.get_logger().warn("Emergency stop requested.")

    def publish_waypoints(self, poses_base: List[List[float]]):
        arr = PoseArray()
        arr.header.frame_id = "world" if self.ik_base_ref == -1 else "base"
        for p in poses_base:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = p[0], p[1], p[2]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = p[3], p[4], p[5], p[6]
            arr.poses.append(pose)
        self.pub_waypoints.publish(arr)

    # -------- main callback --------
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
                R = np.eye(3, dtype=float)
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

        # MAT -> IK-BASE transform (world if ik_base_ref == -1)
        base_T_mat = self.get_pose_rel(self.h_mat, self.ik_base_ref)
        p = np.array(base_T_mat[:3], dtype=float)
        q = base_T_mat[3:7]
        Rb = _R_from_quat(*q)
        T_base_mat = np.eye(4, dtype=float); T_base_mat[:3,:3] = Rb; T_base_mat[:3,3] = p

        poses_base: List[List[float]] = []
        tags: List[str] = []
        for (pxyzw, tag) in seq_mat:
            px,py,pz,qx,qy,qz,qw = pxyzw
            T_mat_tcp = np.eye(4, dtype=float)
            T_mat_tcp[:3,:3] = _R_from_quat(qx,qy,qz,qw)
            T_mat_tcp[:3,3]  = np.array([px,py,pz], dtype=float)
            T_base_tcp = T_base_mat @ T_mat_tcp
            pos = T_base_tcp[:3,3]
            quat = _quat_from_R(T_base_tcp[:3,:3])
            poses_base.append([pos[0],pos[1],pos[2], *quat])
            tags.append(tag)

        # publish for visualization
        self.publish_waypoints(poses_base)
        if self.dry_run:
            self.get_logger().info(f"[DRY] Generated {len(poses_base)} waypoints (peck profile).")
            return

        # execute with simIK solve on each sub-step
        self.get_logger().info(f"Executing {len(poses_base)} waypoints (simIK)...")
        last_pose = self.get_pose_rel(self.h_target, self.ik_base_ref)
        curr_p = np.array(last_pose[:3], dtype=float)
        curr_q = tuple(last_pose[3:7])

        for i, pb in enumerate(poses_base):
            if self._stop_flag:
                self.get_logger().warn("Stopped by /suture_stop")
                break

            next_p = np.array(pb[:3], dtype=float)
            next_q = tuple(pb[3:7])

            for pos, quat in _interp_pose(curr_p, curr_q, next_p, next_q, step_m=self.travel_step):
                if self._stop_flag:
                    break
                self.set_target_pose_and_solve(_pose_to_list(pos, quat))
                if self.dt > 0.0:
                    time.sleep(self.dt)

            if tags[i] == "down" and not self._stop_flag and self.dwell_s > 0.0:
                time.sleep(self.dwell_s)

            curr_p, curr_q = next_p, next_q

        self.get_logger().info("Execution complete.")


# ---------------- entry ----------------
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
