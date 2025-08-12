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
from ikpy.link import URDFLink
# xacro/URDF
import subprocess
from ament_index_python.packages import get_package_share_directory

# CoppeliaSim ZMQ Remote API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    # old import name fallback
    from zmqRemoteApi import RemoteAPIClient


# ----------------- Config -----------------
ROBOT = 'ur3'   # ur5 or ur5e; change URDF path below if needed
ROBOT_BASE_PATH = '/UR3'  # CoppeliaSim object path of the robot base
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

MAT_FRAME_NAME = 'mat'       # logical frame name (used in your ML & transforms)
BASE_FRAME_NAME = 'base_link'  # robot base frame (URDF base)
TOOL_FRAME_NAME = 'tool0'    # UR DF tip frame (UR robots use tool0)

# stitch params (meters)
DEFAULT_SPACING = 0.006   # 6 mm between stitches
DEFAULT_BITE    = 0.005   # 5 mm bite width (entry->exit offset)
DEFAULT_DEPTH   = 0.003   # 3 mm z-depth below mat surface
APPROACH_Z      = 0.012   # 12 mm above mat for approach/retract
LINEAR_STEP     = 0.01    # rad step per joint when streaming to sim
POSE_TOL        = 1e-3    # IK numeric tolerance
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


def plan_stitches_for_cut(polyline_mat: np.ndarray,
                          spacing=DEFAULT_SPACING,
                          bite=DEFAULT_BITE,
                          depth=DEFAULT_DEPTH,
                          approach=APPROACH_Z) -> List[List[Pose]]:
    """
    Given a 2D or 3D polyline in the mat frame, return a list of stitch 'segments'.
    Each segment is [approach, entry, pierce_mid, exit, retract] poses.
    Tool orientation: z-down, yaw tangent to the cut.
    """
    if polyline_mat.shape[1] == 2:
        poly = np.c_[polyline_mat, np.zeros(len(polyline_mat))]
    else:
        poly = polyline_mat.copy()

    points = sample_polyline(poly[:, :2], spacing)  # 2D sampling for tangent
    segs: List[List[Pose]] = []

    for i, p in enumerate(points):
        # tangent at p
        if i == 0:
            t = points[min(1, len(points)-1)] - points[0]
        elif i == len(points)-1:
            t = points[-1] - points[-2]
        else:
            t = points[i+1] - points[i-1]
        if np.linalg.norm(t) < 1e-9:
            continue
        t = t / np.linalg.norm(t)
        # normal to the cut in mat plane (rotate tangent by +90 deg)
        n = np.array([-t[1], t[0]])
        entry_xy = p + (bite/2.0) * n
        exit_xy  = p - (bite/2.0) * n

        # base z on poly's z if provided, otherwise 0 (mat surface)
        z_surface = np.interp(0, [0, 1], [0, 0])  # 0 for now
        entry = np.array([entry_xy[0], entry_xy[1], z_surface - depth])
        exitp = np.array([exit_xy[0],  exit_xy[1],  z_surface - depth])
        approach_p = np.array([entry_xy[0], entry_xy[1], z_surface + approach])
        retract_p  = np.array([exit_xy[0],  exit_xy[1],  z_surface + approach])

        # orientation: z-down, yaw aligned with tangent
        yaw = math.atan2(t[1], t[0])
        rpy = (math.pi, 0.0, yaw)  # roll 180° -> z-down (UR's tool z points out)
        segs.append([
            Pose(approach_p, rpy),
            Pose(entry, rpy),
            Pose((entry + exitp)/2.0, rpy),  # mid pierce (straight approx.)
            Pose(exitp, rpy),
            Pose(retract_p, rpy),
        ])
    return segs


def homogeneous_from_pose(p: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = tft.euler_matrix(*p.rpy)[:3, :3]
    T[:3, 3] = p.xyz
    return T


class IKUR:
    """
    Lightweight IK using UR description with ikpy.
    """

    def __init__(self, robot: str = 'ur3'):
        """
        Expand the UR xacro into URDF, patch 'continuous' joints so ikpy accepts them,
        and build an IK chain that exposes exactly the 6 UR joints (by name).
        """
        import os, re, subprocess, numpy as np
        from ament_index_python.packages import get_package_share_directory
        from ikpy.chain import Chain
        from ikpy.link import URDFLink

        # ---- xacro paths & args ----
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

        # ---- expand xacro ----
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

        # ---- build chain ----
        self.chain = Chain.from_urdf_file(
            tmp_urdf,
            base_elements=[BASE_FRAME_NAME],
            active_links_mask=None
        )

        # ---- select exactly the 6 UR joints by name ----
        # These names must match the URDF joint names (they do for UR family)
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

        # Warm-start vector for IK
        self.q = np.zeros(len(self.chain.links))

    def solve(self, target_T: np.ndarray, q_seed: np.ndarray = None) -> np.ndarray:
        if q_seed is not None:
            # map seed into full vector
            q_full = self.q.copy()
            for k, idx in enumerate(self.active_idx):
                q_full[idx] = q_seed[k]
        else:
            q_full = self.q.copy()

        q_sol = self.chain.inverse_kinematics_frame(
            target_T, initial_position=q_full, orientation_mode='all')  # 6D IK
        # extract 6 joint angles in the same order we detected as active links
        q6 = np.array([q_sol[idx] for idx in self.active_idx])
        # store back to full for warm starting next iteration
        for k, idx in enumerate(self.active_idx):
            self.q[idx] = q6[k]
        return q6


class CoppeliaDriver:
    def __init__(self, joint_names: List[str], robot_base_path: str = '/UR3'):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # # Start the simulation if stopped and enable stepped mode
        # try:
        #     if hasattr(self.sim, 'setStepping'):
        #         self.sim.setStepping(True)
        #     state = self.sim.getSimulationState()
        #     if state == self.sim.simulation_stopped:
        #         self.sim.startSimulation()
        # except Exception as e:
        #     print('[WARN] Could not auto-start simulation:', e)



        # Stepped mode (deterministic stepping)
        if hasattr(self.sim, 'setStepping'):
            self.sim.setStepping(True)

        # ---- Resolve base handle robustly ----
        base_candidates = [robot_base_path]
        if not robot_base_path.endswith('#0'):
            base_candidates.append(robot_base_path + '#0')
        # also try plain alias (without leading '/')
        if robot_base_path.startswith('/'):
            base_candidates.append(robot_base_path[1:])

        base_handle = None
        last_err = None
        for cand in base_candidates:
            try:
                base_handle = self.sim.getObject(cand)
                ROBOT_PATH_USED = cand
                break
            except Exception as e:
                last_err = e
        if base_handle is None:
            raise RuntimeError(
                f"Could not find robot base at any of: {base_candidates}. "
                f"Rename your root object in Coppelia or update ROBOT_BASE_PATH. "
                f"Last error: {last_err}"
            )

        # ---- Resolve joints (try absolute path first, then global alias) ----
        self.joint_handles = []
        missing = []
        for jn in joint_names:
            h = None
            # try as child path of the base
            for cand in (f'{ROBOT_PATH_USED}/{jn}', jn):
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
            # Try to help: list all joints under the base so you can see actual names
            try:
                JOINT = getattr(self.sim, 'object_joint_type', 2)  # fallback enum value
                under_base = self.sim.getObjectsInTree(base_handle, JOINT, 1)  # include base's children
                aliases = [self.sim.getObjectAlias(h, 1) for h in under_base]
            except Exception:
                aliases = ['(could not query aliases)']

            raise RuntimeError(
                "Could not find these joint(s): "
                + ", ".join(missing)
                + f"\nLook under robot base '{ROBOT_PATH_USED}' and rename them to match, or update JOINT_NAMES.\n"
                + "Joints found under the base were:\n  - "
                + "\n  - ".join(aliases)
            )

        # Optional tool joint
        try:
            self.tool_handle = self.sim.getObject(f'{ROBOT_PATH_USED}/tool_opening_joint')
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

        # For simplicity assume mat frame == robot base XY plane at z=0.
        # If you have a calibrated transform T_base_mat, apply it here to lift poses into base_link.
        T_base_mat = np.eye(4)

        for cut in data['cuts']:
            poly = np.array(cut['polyline'], dtype=float)  # Nx2 or Nx3 in meters
            segments = plan_stitches_for_cut(poly, spacing, bite, depth, approach=APPROACH_Z)
            self.execute_segments(segments, T_base_mat)

        self.get_logger().info('Finished suturing all cuts.')

    def execute_segments(self, segments: List[List[Pose]], T_base_mat: np.ndarray):
        q_seed = self.driver.get_joints()
        # slight tool open (if present)
        self.driver.set_tool(0.0)

        for seg in segments:
            for pose in seg:
                # Lift pose from mat frame into base_link
                T = T_base_mat @ homogeneous_from_pose(pose)
                q_target = self.ik.solve(T, q_seed)
                self.driver.goto(q_target)
                q_seed = q_target.copy()
        # retract tool a bit open/close if you want
        self.driver.set_tool(0.0)


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
