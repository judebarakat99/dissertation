#!/usr/bin/env python3
import os
import json
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf_transformations as tft

# IK
from ikpy.chain import Chain
# xacro/URDF
import subprocess
from ament_index_python.packages import get_package_share_directory

# CoppeliaSim ZMQ Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    from zmqRemoteApi import RemoteAPIClient  # legacy fallback


# ----------------- Config -----------------
ROBOT = 'ur3'                # 'ur3', 'ur3e', 'ur5', 'ur5e', ...
ROBOT_BASE_PATH = '/UR3'     # CoppeliaSim object path of the robot base
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',           # you said you've renamed it to elbow_joint
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

MAT_FRAME_NAME  = 'mat'         # dummy or shape alias in your scene
BASE_FRAME_NAME = 'base_link'   # URDF base link
TOOL_FRAME_NAME = 'tool0'       # UR tool frame

# stitch params (meters)
DEFAULT_SPACING = 0.006   # 6 mm between stitches
DEFAULT_BITE    = 0.005   # 5 mm bite width (entry->exit offset across cut)
DEFAULT_DEPTH   = 0.003   # 3 mm z-depth below mat surface
APPROACH_Z      = 0.012   # 12 mm approach/retract height
LINEAR_STEP     = 0.01    # rad step per joint when streaming to sim
POSE_TOL        = 1e-3
N_PIERCE_PTS    = 7
SINGULARITY_TILT = 0.20   # ~11.5°, small pitch to avoid straight-wrist singularity

TWOPI = 2.0 * math.pi

# Prefer an "elbow-up" comfortable posture (UR3-ish) for IK seeding
NOMINAL_Q = np.array([0.0, -1.35, 1.90, 0.0, 1.30, 0.0])  # radians

# Soft joint limits to keep safe configurations (adjust if needed)
#          J1 (pan)      J2 (shoulder)  J3 (elbow)    J4 (wrist1)  J5 (wrist2)  J6 (wrist3)
SOFT_LIMITS = np.array([
    [-math.pi,  math.pi],   # [-180°, 180°]
    [-2.60,     -0.10],     # keep shoulder down-ish
    [ 0.00,      3.00],     # elbow mostly in front
    [-2.50,      2.50],     # avoid extreme wrist1 bend
    [ 0.25,      2.80],     # avoid straight wrist2 (~0) & extremes
    [-math.pi,  math.pi],
], dtype=float)
# ------------------------------------------


@dataclass
class Pose:
    xyz: np.ndarray  # (3,)
    rpy: Tuple[float, float, float]  # roll, pitch, yaw


def rpy_to_quat(rpy):
    return tft.quaternion_from_euler(rpy[0], rpy[1], rpy[2])


def sample_polyline(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline [N x 2] (or [N x 3]) at approx. equal distance."""
    dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(poly, axis=0), axis=1)])
    if dists[-1] < 1e-9:
        return poly[:1]
    s = np.arange(0, dists[-1], spacing)
    resampled = np.vstack([np.interp(s, dists, poly[:, i]) for i in range(poly.shape[1])]).T
    if (dists[-1] - s[-1]) > 1e-6:
        resampled = np.vstack([resampled, poly[-1]])
    return resampled


def bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> List[np.ndarray]:
    """Quadratic Bézier curve points from p0 → p2 with control p1."""
    ts = np.linspace(0.0, 1.0, n)
    return [((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2 for t in ts]


def wrap_to_nearest(q_target: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Add/subtract 2π to each joint so it is nearest to q_ref (prevents multi-turn spins)."""
    q_out = q_target.copy()
    for i in range(len(q_out)):
        d = q_out[i] - q_ref[i]
        d = (d + math.pi) % TWOPI - math.pi  # wrap to (-pi, pi]
        q_out[i] = q_ref[i] + d
    return q_out


def clamp_soft_limits(q: np.ndarray) -> np.ndarray:
    """Clamp joint angles to the configured soft limits."""
    q = q.copy()
    for i in range(6):
        lo, hi = SOFT_LIMITS[i]
        q[i] = min(max(q[i], lo), hi)
    return q


def blend_seed(q_seed: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Bias IK initial guess toward a nominal elbow-up posture."""
    return alpha * q_seed + (1.0 - alpha) * NOMINAL_Q


def plan_continuous_stitch(polyline_mat: np.ndarray,
                           spacing=DEFAULT_SPACING,
                           bite=DEFAULT_BITE,
                           depth=DEFAULT_DEPTH,
                           approach=APPROACH_Z,
                           clearance=0.006,
                           n_pierce_pts=N_PIERCE_PTS) -> List[List[Pose]]:
    """
    Continuous running suture: approach once, then for each stitch do
    entry -> curved pierce -> exit -> small clearance hop -> next entry,
    and a final retract at the end. Returns [ [Pose,...] ] (one long segment).
    """
    if polyline_mat.shape[1] == 2:
        poly = np.c_[polyline_mat, np.zeros(len(polyline_mat))]
    else:
        poly = polyline_mat.copy()

    pts2d = sample_polyline(poly[:, :2], spacing)
    if len(pts2d) == 0:
        return [[]]

    path: List[Pose] = []
    z_surface = 0.0

    for i, p in enumerate(pts2d):
        # tangent
        if i == 0:
            t = pts2d[min(1, len(pts2d)-1)] - pts2d[0]
        elif i == len(pts2d)-1:
            t = pts2d[-1] - pts2d[-2]
        else:
            t = pts2d[i+1] - pts2d[i-1]
        if np.linalg.norm(t) < 1e-9:
            continue
        t = t / (np.linalg.norm(t) + 1e-9)
        n = np.array([-t[1], t[0]])

        entry_xy = p + (bite/2.0) * n
        exit_xy  = p - (bite/2.0) * n

        entry  = np.array([entry_xy[0], entry_xy[1], z_surface - depth])
        exitp  = np.array([exit_xy[0],  exit_xy[1],  z_surface - depth])
        appr   = np.array([entry_xy[0], entry_xy[1], z_surface + approach])
        clearE = np.array([exit_xy[0],  exit_xy[1],  z_surface + clearance])

        yaw = math.atan2(t[1], t[0])
        rpy = (math.pi, -SINGULARITY_TILT, yaw)  # z-down with a small pitch tilt

        # first stitch: approach once
        if i == 0:
            path.append(Pose(appr, rpy))

        # curved pierce via quadratic Bézier
        mid = 0.5 * (entry + exitp)
        extra_down = max(0.5 * bite, 0.8 * depth)
        ctrl = mid + np.array([0.0, 0.0, -extra_down])
        for q in bezier_curve(entry, ctrl, exitp, n_pierce_pts):
            path.append(Pose(q, rpy))

        # small clearance hop after each exit (except last we'll retract higher)
        if i < len(pts2d) - 1:
            path.append(Pose(clearE, rpy))
            # small lateral move at clearance height toward next entry
            p_next = pts2d[i+1]
            t_next = (pts2d[min(i+2, len(pts2d)-1)] - p) if i < len(pts2d)-2 else (p_next - p)
            t_next = t_next / (np.linalg.norm(t_next) + 1e-9)
            n_next = np.array([-t_next[1], t_next[0]])
            next_entry_xy = p_next + (bite/2.0) * n_next
            next_entry_clear = np.array([next_entry_xy[0], next_entry_xy[1], z_surface + clearance])
            path.append(Pose(next_entry_clear, rpy))
        else:
            # last stitch: retract to approach height
            retract = np.array([exit_xy[0], exit_xy[1], z_surface + approach])
            path.append(Pose(retract, rpy))

    # return as one long segment for execute()
    return [path]


def homogeneous_from_pose(p: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = tft.euler_matrix(*p.rpy)[:3, :3]
    T[:3, 3] = p.xyz
    return T


# ---------- Simple drawing helpers (overlay in CoppeliaSim) ----------
def draw_reset(sim, handles: dict):
    for h in list(handles.values()):
        try:
            sim.removeDrawingObject(h)
        except Exception:
            pass
    handles.clear()

def draw_cut_and_stitches(sim, base_handle, T_base_mat, polyline_mat: np.ndarray, poses: List[Pose], handles: dict):
    """Draws: green line for cut, small red spheres for stitch samples."""
    if 'cut' not in handles:
        handles['cut'] = sim.addDrawingObject(sim.drawing_lines, 2.0, 0.0, base_handle, 10000, [0,1,0])
    if 'pts' not in handles:
        handles['pts'] = sim.addDrawingObject(sim.drawing_spherepoints, 0.006, 0.0, base_handle, 10000, [1,0,0])

    def mat_to_base(p3):
        p = np.r_[p3, 1.0]
        q = (T_base_mat @ p)[:3]
        return q.tolist()

    # cut line
    if polyline_mat.shape[1] == 2:
        poly3 = np.c_[polyline_mat, np.zeros(len(polyline_mat))]
    else:
        poly3 = polyline_mat
    for a, b in zip(poly3[:-1], poly3[1:]):
        sim.addDrawingObjectItem(handles['cut'], mat_to_base(a) + mat_to_base(b))

    # stitch points
    for P in poses:
        sim.addDrawingObjectItem(handles['pts'], P.xyz.tolist())
# --------------------------------------------------------------------


class IKUR:
    """
    Lightweight IK using UR description with ikpy.
    """
    def __init__(self, robot: str = 'ur3'):
        """
        Expand the UR xacro into URDF, patch 'continuous' joints so ikpy accepts them,
        and build an IK chain that exposes exactly the 6 UR joints (by name).
        """
        import re

        ur_desc = get_package_share_directory('ur_description')
        xacro_path = os.path.join(ur_desc, 'urdf', 'ur.urdf.xacro')

        valid = {'ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e', 'ur16e', 'ur20'}
        ur_type = robot.lower()
        if ur_type not in valid:
            raise ValueError(f"Unsupported UR type '{robot}'. Choose one of: {sorted(valid)}")
        cfg_dir = os.path.join(ur_desc, 'config', ur_type)

        xacro_cmd = [
            'xacro', xacro_path,
            'name:=ur',
            f'ur_type:={ur_type}',
            f'kinematics_params:={os.path.join(cfg_dir, "default_kinematics.yaml")}',
            f'joint_limit_params:={os.path.join(cfg_dir, "joint_limits.yaml")}',
            f'physical_params:={os.path.join(cfg_dir, "physical_parameters.yaml")}',
            f'visual_params:={os.path.join(cfg_dir, "visual_parameters.yaml")}',
        ]

        urdf_xml = subprocess.check_output(xacro_cmd)

        # ---- patch for ikpy ----
        txt = urdf_xml.decode('utf-8')
        txt = txt.replace('type="continuous"', 'type="revolute"')

        def add_limit_if_missing(m):
            block = m.group(0)
            if '<limit' in block:
                return block
            return block.replace(
                '</joint>',
                '<limit lower="-6.283185307179586" upper="6.283185307179586" velocity="3.0" effort="50.0"/></joint>'
            )

        txt = re.sub(r'<joint\b[^>]*type="revolute"[^>]*>.*?</joint>',
                     add_limit_if_missing, txt, flags=re.DOTALL)

        tmp_urdf = '/tmp/_ur.urdf'
        with open(tmp_urdf, 'w') as f:
            f.write(txt)

        self.chain = Chain.from_urdf_file(
            tmp_urdf,
            base_elements=[BASE_FRAME_NAME],
            active_links_mask=None
        )

        target_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        link_names = [getattr(l, 'name', f'link_{i}') for i, l in enumerate(self.chain.links)]
        name_to_idx = {n: i for i, n in enumerate(link_names)}

        missing = [n for n in target_joint_names if n not in name_to_idx]
        if missing:
            raise RuntimeError("Could not find these UR joints in the URDF: " + ", ".join(missing)
                               + f"\nAvailable links were: {link_names}")

        self.active_idx = [name_to_idx[n] for n in target_joint_names]
        self.q = np.zeros(len(self.chain.links))  # warm-start vector

    def solve(self, target_T: np.ndarray, q_seed: np.ndarray = None) -> np.ndarray:
        # initial guess: blend last command with a nominal elbow-up posture
        if q_seed is None:
            q_seed = NOMINAL_Q
        q_init_full = self.q.copy()
        for k, idx in enumerate(self.active_idx):
            q_init_full[idx] = blend_seed(q_seed)[k]

        # 1) Try strict 6D orientation
        try:
            q_sol_all = self.chain.inverse_kinematics_frame(
                target_T, initial_position=q_init_full, orientation_mode='all')
            q6 = np.array([q_sol_all[idx] for idx in self.active_idx])
            q6 = clamp_soft_limits(q6)
            for k, idx in enumerate(self.active_idx):
                self.q[idx] = q6[k]
            return q6
        except Exception:
            pass

        # 2) Fallback: align only tool Z (lets yaw/roll vary if needed)
        q_sol_z = self.chain.inverse_kinematics_frame(
            target_T, initial_position=q_init_full, orientation_mode='z')
        q6 = np.array([q_sol_z[idx] for idx in self.active_idx])
        q6 = clamp_soft_limits(q6)
        for k, idx in enumerate(self.active_idx):
            self.q[idx] = q6[k]
        return q6


class CoppeliaDriver:
    def __init__(self, joint_names: List[str], robot_base_path: str = '/UR3'):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # Stepped mode + make sure the sim is running
        if hasattr(self.sim, 'setStepping'):
            self.sim.setStepping(True)
        try:
            state = self.sim.getSimulationState()
            if state in (self.sim.simulation_stopped, self.sim.simulation_paused):
                self.sim.startSimulation()
        except Exception as e:
            print('[WARN] Could not auto-start simulation:', e)

        # ---- Resolve base handle robustly ----
        base_candidates = [robot_base_path]
        if not robot_base_path.endswith('#0'):
            base_candidates.append(robot_base_path + '#0')
        if robot_base_path.startswith('/'):
            base_candidates.append(robot_base_path[1:])

        self.base_handle = None
        self.robot_path_used = None
        last_err = None
        for cand in base_candidates:
            try:
                self.base_handle = self.sim.getObject(cand)
                self.robot_path_used = cand
                break
            except Exception as e:
                last_err = e
        if self.base_handle is None:
            raise RuntimeError(
                f"Could not find robot base at any of: {base_candidates}. "
                f"Rename your root object in Coppelia or update ROBOT_BASE_PATH. "
                f"Last error: {last_err}"
            )

        # ---- Resolve joints ----
        self.joint_handles = []
        missing = []
        for jn in joint_names:
            h = None
            for cand in (f'{self.robot_path_used}/{jn}', jn):
                try:
                    h = self.sim.getObject(cand)
                    break
                except Exception:
                    pass
            if h is None:
                missing.append(jn)
            else:
                self.joint_handles.append(h)

        if missing:
            try:
                JOINT = getattr(self.sim, 'object_joint_type', 2)
                under_base = self.sim.getObjectsInTree(self.base_handle, JOINT, 1)
                aliases = [self.sim.getObjectAlias(h, 1) for h in under_base]
            except Exception:
                aliases = ['(could not query aliases)']

            raise RuntimeError(
                "Could not find these joint(s): "
                + ", ".join(missing)
                + f"\nLook under robot base '{self.robot_path_used}' and rename them to match, or update JOINT_NAMES.\n"
                + "Joints found under the base were:\n  - "
                + "\n  - ".join(aliases)
            )

        # Optional tool joint
        try:
            self.tool_handle = self.sim.getObject(f'{self.robot_path_used}/tool_opening_joint')
        except Exception:
            try:
                self.tool_handle = self.sim.getObject('tool_opening_joint')
            except Exception:
                self.tool_handle = None

    def get_joints(self) -> np.ndarray:
        return np.array([self.sim.getJointPosition(h) for h in self.joint_handles])

    def set_tool(self, opening: float):
        if self.tool_handle is not None:
            self.sim.setJointTargetPosition(self.tool_handle, float(opening))

    def goto(self, q_target: np.ndarray, step=LINEAR_STEP, settle_time=0.05):
        q_curr = self.get_joints()
        # shortest-path + soft limits
        q_target = wrap_to_nearest(q_target, q_curr)
        q_target = clamp_soft_limits(q_target)

        delta = q_target - q_curr
        steps = max(1, int(np.max(np.abs(delta)) / step))
        for s in range(1, steps + 1):
            q = q_curr + (s / steps) * delta
            for h, qi in zip(self.joint_handles, q):
                self.sim.setJointTargetPosition(h, float(qi))
            self.sim.step()
        time.sleep(settle_time)


class SutureNode(Node):
    """
    Subscribes to /suture_cuts (std_msgs/String with JSON payload),
    plans stitches, runs IK, and streams joints to CoppeliaSim.
    """
    def __init__(self):
        super().__init__('suture_arm')
        self.sub = self.create_subscription(String, '/suture_cuts', self.on_cuts, 10)
        self.ik = IKUR(ROBOT)
        self.driver = CoppeliaDriver(JOINT_NAMES, ROBOT_BASE_PATH)

        # --- Calibrate T_base_mat from the scene (mat relative to robot base) ---
        self.T_base_mat = np.eye(4)
        try:
            mat_candidates = [f'/{MAT_FRAME_NAME}', MAT_FRAME_NAME, f'/{MAT_FRAME_NAME}#0']
            mat_h = None
            for cand in mat_candidates:
                try:
                    mat_h = self.driver.sim.getObject(cand)
                    break
                except Exception:
                    pass
            if mat_h is None:
                self.get_logger().warn(f"Mat '{MAT_FRAME_NAME}' not found. Using identity T_base_mat.")
            else:
                pos = self.driver.sim.getObjectPosition(mat_h, self.driver.base_handle)     # [x,y,z]
                rpy = self.driver.sim.getObjectOrientation(mat_h, self.driver.base_handle)  # [roll,pitch,yaw]
                R = tft.euler_matrix(*rpy)[:3, :3]
                self.T_base_mat[:3, :3] = R
                self.T_base_mat[:3,  3] = np.array(pos)
                self.get_logger().info(f"T_base_mat set from scene. pos={pos}, rpy={rpy}")
        except Exception as e:
            self.get_logger().warn(f"Failed to compute T_base_mat from scene: {e}. Using identity.")

        self._draw = {}  # drawing object handles
        self.get_logger().info('SutureNode ready. Waiting for /suture_cuts ...')

    def on_cuts(self, msg: String):
        """
        Expected JSON schema (meters):
        {
          "frame_id": "mat",
          "cuts": [
             { "polyline": [[x,y],[x,y],...]} , ...
          ],
          "params": {"spacing":0.008,"bite":0.006,"depth":0.004}
        }
        """
        data = json.loads(msg.data)
        spacing = data.get('params', {}).get('spacing', DEFAULT_SPACING)
        bite    = data.get('params', {}).get('bite',    DEFAULT_BITE)
        depth   = data.get('params', {}).get('depth',   DEFAULT_DEPTH)

        self.get_logger().info(f"Got /suture_cuts: {len(data.get('cuts', []))} cut(s) "
                               f"(spacing={spacing:.3f}, bite={bite:.3f}, depth={depth:.3f})")

        for cut in data['cuts']:
            poly = np.array(cut['polyline'], dtype=float)  # Nx2 or Nx3 in meters
            segments = plan_continuous_stitch(
                poly, spacing, bite, depth, approach=APPROACH_Z, clearance=0.006
            )
            all_poses = [p for seg in segments for p in seg]

            # draw overlays (cut + planned path points)
            draw_reset(self.driver.sim, self._draw)
            draw_cut_and_stitches(self.driver.sim, self.driver.base_handle, self.T_base_mat, poly, all_poses, self._draw)

            self.get_logger().info(f"Planned {len(all_poses)} poses for this cut")
            self.execute_segments(segments)

        self.get_logger().info('Finished suturing all cuts.')

    def execute_segments(self, segments: List[List[Pose]]):
        q_seed = self.driver.get_joints()
        self.driver.set_tool(0.02)  # slightly open (if present)

        for seg in segments:
            for pose in seg:
                T = self.T_base_mat @ homogeneous_from_pose(pose)  # lift from mat to base_link
                q_target = self.ik.solve(T, q_seed)
                # keep near previous branch + within soft limits
                q_target = wrap_to_nearest(q_target, q_seed)
                q_target = clamp_soft_limits(q_target)
                self.driver.goto(q_target)
                q_seed = q_target.copy()

        self.driver.set_tool(0.02)  # open at the end (if present)


def main():
    rclpy.init()
    node = SutureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
