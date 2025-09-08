# dissertation
CV_cut_identification is to try to identify the cuts on the suture mat using the realsense camera through cv code

yolov8 has my attempt at using yolov8 to identify wounds. create_dataset is to create a dataset of the suture pad under different lightings to be used for ML code

my_ML is a complete pipeline for training, validating, and testing an object detection model using PyTorch and the Faster R-CNN architecture on a COCO-formatted custom dataset.

Laparoscopic Suturing with UR3 in CoppeliaSim (ROS2 Humble)
===========================================================

This project integrates a UR3 robotic arm in CoppeliaSim with ROS2 nodes for laparoscopic suturing experiments. 
It combines simulation, computer vision, and inverse kinematics to reproduce suturing motions on a synthetic mold.


Repository Structure
--------------------

ros_ws2/src
├── suture_arm
│   ├── launch/                 # ROS2 launch files
│   ├── ML_detection/           # Machine learning models for cut detection
│   ├── resource/               # STL models, scene files, and documentation
│   ├── suture_arm/             # Main ROS2 package Python nodes
│   │   ├── laparoscopic_ik_node.py   # IK control for UR3 needle tip
│   │   ├── vision_web.py             # Vision + web interface node
│   │   ├── stitching.py              # Stitching logic
│   │   └── (utilities and dataset scripts)
│   ├── templates/              # Web UI templates
│   └── test/                   # Code style & testing
└── ur_description              # UR robot descriptions, meshes, and configs


Prerequisites
-------------

- Ubuntu 22.04 with ROS2 Humble installed
- CoppeliaSim (latest version)
- Python dependencies (install with pip):

  pip install numpy opencv-python torch torchvision

- Clone/download the ML detection models folder from:
  https://livemanchesterac-my.sharepoint.com/:f:/g/personal/jude_barakat_postgrad_manchester_ac_uk/EiiYwXlPeD1FgRe4GYUI2OoB5fPw_mqk-WBb6kzutxiBEQ?e=9peuTS

Place the downloaded contents inside:
suture_arm/ML_detection/


Running the System
------------------

1. Open the simulation
   Launch ur3_suture.ttt in CoppeliaSim:
   ros_ws2/src/suture_arm/resource/ur3_suture.ttt

2. Build the workspace
   cd ~/Documents/Dissertation/Code/dissertation/ROS/attempt4/ros_ws2
   colcon build

3. Source and run vision system (first terminal)
   source install/setup.bash
   ros2 run suture_arm vision_web

4. Source and run IK controller (second terminal)
   source install/setup.bash
   ros2 run suture_arm laparoscopic_ik_node


Components
----------

- vision_web
  Publishes target stitching points from camera input via ML cut detection.

- laparoscopic_ik_node
  Drives the UR3 to align the needle tip with targets using damped-least-squares IK.

- stitching
  Handles stitching path generation and execution.

- ur_description
  Contains UR robot description files (meshes, configs, launch files).


Notes
-----

- Ensure CoppeliaSim remote API is enabled (ZMQ API).
- If the robot collides or freezes, restart both CoppeliaSim and ROS nodes.
- For testing, you can run the demo launch:

  ros2 launch suture_arm suture_demo.launch.py


License
-------

This project builds upon open-source packages. Check individual subfolders for license details.

