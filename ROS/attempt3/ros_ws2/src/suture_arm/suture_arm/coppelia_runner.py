#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Optional
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

DEFAULT_SCENE_NAME = "ur3_suture.ttt"

def _exists(p: Optional[str]) -> Optional[str]:
    return p if (p and os.path.isfile(p)) else None

def _find_coppelia_bin() -> Optional[str]:
    # 1) explicit env overrides
    for k in ("SUTURE_ARM_COPPELIA_BIN", "COPPELIA_BIN"):
        b = _exists(os.getenv(k))
        if b: return b
    root = os.getenv("COPPELIA_SIM_ROOT")
    if root:
        for cand in ("coppeliaSim.sh", "coppeliaSim"):  # AppImage or shell
            b = _exists(os.path.join(root, cand))
            if b: return b

    # 2) PATH
    for exe in ("coppeliaSim.sh", "coppeliaSim"):
        b = shutil.which(exe)
        if b: return b

    # 3) common install locations
    common = [
        "/usr/local/coppeliaSim/coppeliaSim.sh",
        "/opt/CoppeliaSim/coppeliaSim.sh",
        os.path.expanduser("~/CoppeliaSim/coppeliaSim.sh"),
        os.path.expanduser("~/Programs/CoppeliaSim/coppeliaSim.sh"),
    ]
    for p in common:
        if os.path.isfile(p): return p
    return None

def _find_scene_file(scene_param: Optional[str], pkg_name="suture_arm") -> Optional[str]:
    """
    Try (in order):
      - exact path if it exists
      - package share: <share>/resource/<name>
      - source tree  : <src_dir>/../resource/<name>
    """
    name = scene_param or DEFAULT_SCENE_NAME
    # allow giving just a stem
    if not os.path.splitext(name)[1]:
        name = name + ".ttt"

    # 1) exact path
    if os.path.isabs(name) and os.path.isfile(name):
        return name

    # 2) package share
    try:
        share = get_package_share_directory(pkg_name)
        cand = os.path.join(share, "resource", name)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass

    # 3) source tree fallback (works even if not installed into share/)
    here = os.path.dirname(__file__)
    src_cand = os.path.abspath(os.path.join(here, "..", "resource", name))
    if os.path.isfile(src_cand):
        return src_cand

    # 4) one more: relative to current working directory
    cwd_cand = os.path.abspath(os.path.join(os.getcwd(), name))
    if os.path.isfile(cwd_cand):
        return cwd_cand

    return None


class CoppeliaRunner(Node):
    def __init__(self):
        super().__init__("coppelia_runner")

        self.declare_parameter("coppelia_bin", "")
        self.declare_parameter("scene", DEFAULT_SCENE_NAME)
        self.declare_parameter("headless", True)
        self.declare_parameter("wait", False)          # keep node alive after launch
        self.declare_parameter("extra", "")
        self.declare_parameter("zmq_port", 23000)

        # Resolve binary
        user_bin = self.get_parameter("coppelia_bin").get_parameter_value().string_value.strip()
        coppelia_bin = _exists(user_bin) or _find_coppelia_bin()
        if not coppelia_bin:
            self.get_logger().error("coppelia_bin not set and auto-detect failed "
                                    "(set param 'coppelia_bin' or env COPPELIA_BIN/COPPELIA_SIM_ROOT).")
            # exit gracefully so launch doesn’t hang
            rclpy.get_default_context().shutdown()
            return

        # Resolve scene
        scene_param = self.get_parameter("scene").get_parameter_value().string_value.strip()
        scene_file = _find_scene_file(scene_param)
        if not scene_file:
            self.get_logger().error(
                f"scene_path must point to an existing .ttt scene "
                f"(tried '{scene_param}', share & source fallbacks)."
            )
            rclpy.get_default_context().shutdown()
            return

        headless = bool(self.get_parameter("headless").get_parameter_value().bool_value)
        wait = bool(self.get_parameter("wait").get_parameter_value().bool_value)
        extra = self.get_parameter("extra").get_parameter_value().string_value or ""
        zmq_port = int(self.get_parameter("zmq_port").get_parameter_value().integer_value)

        args = [coppelia_bin, f"-GzmqRemoteApi.port={zmq_port}"]
        if headless:
            args.append("-h")
        if extra.strip():
            args.extend(shlex.split(extra))
        args.append(scene_file)

        self.get_logger().info(f"Launching CoppeliaSim: {' '.join(shlex.quote(a) for a in args)}")
        try:
            self.proc = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            self.get_logger().error(f"failed to start CoppeliaSim: {e}")
            rclpy.get_default_context().shutdown()
            return

        # If not waiting, just keep node alive and monitor child occasionally
        self.timer = self.create_timer(0.5, self._poll)

        if wait:
            # Block here until the sim exits
            try:
                code = self.proc.wait()
                self.get_logger().info(f"CoppeliaSim exited with code {code}")
            finally:
                rclpy.get_default_context().shutdown()

    def _poll(self):
        if self.proc.poll() is not None:
            self.get_logger().info(f"CoppeliaSim exited with code {self.proc.returncode}")
            # small delay so launch prints nicely, then shutdown node
            time.sleep(0.1)
            rclpy.get_default_context().shutdown()


def main():
    rclpy.init()
    node = CoppeliaRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if getattr(node, "proc", None) and node.proc.poll() is None:
                node.get_logger().info("Terminating CoppeliaSim...")
                node.proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
