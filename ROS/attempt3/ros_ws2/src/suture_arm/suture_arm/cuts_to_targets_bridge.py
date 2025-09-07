#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge: /suture_cuts  -->  /vision_targets

- IN:  std_msgs/String JSON from vision_web.move_robot():
      {"frame_id":"mat", "cuts":[{"polyline":[[x,y], ...]}], "params":{...}}
- OUT: std_msgs/String JSON for laparoscopic_ik_node:
      {"frame":"ur3", "targets":[{"pos":[x,y,z]}, ...]}

Notes:
- Z is filled from a ROS2 param (work_z); default matches laparoscopic_ik_node's idea of the work plane.
- 'frame' is set to "ur3" so the IK node transforms directly into the UR3 base frame.
"""

import json
from typing import List
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CutsToTargetsBridge(Node):
    def __init__(self):
        super().__init__('cuts_to_targets_bridge')

        # Params
        self.declare_parameter('work_z', 0.150)   # meters; keep consistent with laparoscopic_ik_node default
        self.declare_parameter('out_frame', 'ur3')# "ur3" or "scene" (what laparoscopic_ik_node understands)
        self.declare_parameter('throttle_every', 1) # publish every N points (1 = every point)

        self.work_z = float(self.get_parameter('work_z').value)
        self.out_frame = str(self.get_parameter('out_frame').value).lower()
        self.throttle_every = int(self.get_parameter('throttle_every').value)

        self.sub = self.create_subscription(String, '/suture_cuts', self.on_cuts, 10)
        self.pub = self.create_publisher(String, '/vision_targets', 10)

        self.get_logger().info(
            f"cuts_to_targets_bridge running: work_z={self.work_z:.3f} m, frame='{self.out_frame}', "
            f"throttle_every={self.throttle_every}"
        )

    def on_cuts(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse /suture_cuts JSON: {e}")
            return

        cuts = payload.get('cuts') or []
        if not cuts:
            self.get_logger().warn("No 'cuts' in /suture_cuts payload.")
            return

        # Take the first cut's polyline (vision_web currently publishes exactly one)
        poly = cuts[0].get('polyline') or []
        if not poly:
            self.get_logger().warn("Empty 'polyline' in /suture_cuts payload.")
            return

        # Optional decimation/throttling
        if self.throttle_every > 1:
            poly = [pt for i, pt in enumerate(poly) if (i % self.throttle_every) == 0]
            if (len(poly) == 0) and (len(cuts[0].get('polyline', [])) > 0):
                poly = [cuts[0]['polyline'][-1]]  # ensure at least last point

        # Build targets for the IK node: each [x,y] -> {pos:[x,y,work_z]}
        targets = [ {"pos": [float(x), float(y), float(self.work_z)]} for (x, y) in poly ]

        out = {
            "frame": self.out_frame,  # laparoscopic_ik_node understands "ur3" and "scene"
            "targets": targets
        }

        out_msg = String()
        out_msg.data = json.dumps(out)
        self.pub.publish(out_msg)
        self.get_logger().info(f"Bridged {len(targets)} targets to /vision_targets.")

def main():
    rclpy.init()
    node = CutsToTargetsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
