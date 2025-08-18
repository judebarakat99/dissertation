Here’s a drop-in **README.md** you can add to your repo. It documents the whole pipeline (UR3 + CoppeliaSim + ROS 2 + vision web UI + models), the exact object names, env-vars, and common fixes.

---

# UR3 Suturing (ROS 2 Humble + CoppeliaSim, no MoveIt)

This project drives a **UR3** arm in **CoppeliaSim** to execute a simple **continuous suture** over detected cut lines, without MoveIt.
It also hosts a **web dashboard** that streams a top-down **vision sensor** view and overlays the stitched path from multiple ML models.

> Tested on Ubuntu 22.04 · ROS 2 **Humble** · CoppeliaSim **4.1** (ZMQ Remote API).

---

## Contents

* `suture_arm/` (ROS 2 ament\_python package)

  * `suture_arm_node.py` – planner + IK + CoppeliaSim driver
  * `vision_web.py` – Flask web server (MJPEG) with model overlays
  * `ML_detection/` – your `.pth` models + tiny wrapper `.py` files
  * `resource/ur3_suture.ttt` – example scene (optional)
* `ur_description/` – vendor URDF/xacro for UR family (submodule/clone)

---

## Prerequisites

* ROS 2 **Humble**
* Python 3.10
* CoppeliaSim **Edu/Pro 4.1+**

  * **Add-ons → ZeroMQ Remote API server** (enable; set **rpcPort**)
* Packages:

  ```bash
  sudo apt install ros-humble-xacro
  python3 -m pip install --user ikpy tf_transformations numpy opencv-python flask \
      coppeliasim-zmqremoteapi-client
  ```

---

## Workspace Setup

```bash
# create or reuse a ROS 2 ws
mkdir -p ~/Documents/Dissertation/Code/dissertation/ROS/ros_ws2/src
cd ~/Documents/Dissertation/Code/dissertation/ROS/ros_ws2/src

# your package(s)
# (Place this repo's 'suture_arm' folder here)

# UR description (for URDF/xacro)
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git ur_description

# build
cd ..
source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build
source install/setup.bash
```

Sanity checks:

```bash
ros2 pkg executables suture_arm          # expect: suturing, vision_web
ros2 pkg prefix suture_arm               # expect: .../ros_ws2/install/suture_arm
```

---

## CoppeliaSim Scene Setup

1. **Insert UR3**

   * *Models ▸ robots ▸ non-mobile ▸ UR3*.
   * Set **Position** control for all joints; reasonable vel/accel (e.g. 80°/s).
   * **Rename joints** to exactly:

     ```
     shoulder_pan_joint
     shoulder_lift_joint
     elbow_joint
     wrist_1_joint
     wrist_2_joint
     wrist_3_joint
     ```
   * **Rename robot base** object to `UR3` (alias `/UR3`).

2. **Suture pad (mat)**

   * Import your pad mesh (e.g., `suture_pad.stl`), scale to \~**120×80×5 mm**.
   * Create a **dummy** named `mat` located on the pad surface center.
   * Parent the pad shape to the dummy (so `/mat` frame moves with it).

3. **Vision sensor**

   * Add **Vision sensor** named `visionSensor`.
   * Place \~**1 m** above the pad, pointing down; set **Perspective** mode.
   * Resolution **640×480** (or higher), color on.

4. **ZMQ server**

   * **Add-ons → ZeroMQ remote API server** → **Start**.
     Note the port (e.g., **23001**) and enable **Auto-start** if available.

5. **Run the scene** (▶ Play).

> Object names are **case-sensitive** and must match: `/UR3`, `/visionSensor`, `/mat`.

---

## Running the Web Dashboard

The web server needs **no CLI args**. It discovers models and wrappers automatically.

```bash
# if your CoppeliaSim ZMQ runs on port 23001:
CSIM_PORT=23001 ros2 run suture_arm vision_web
```

Open: **[http://localhost:8000/](http://localhost:8000/)**

You’ll see:

* **Raw** live feed from `/visionSensor`
* One card per model overlay (up to 5 by default)
* `/models` page lists detected `.pth` + `.py` pairs

### Configure via environment variables

| Env var         | Default         | Meaning                        |
| --------------- | --------------- | ------------------------------ |
| `CSIM_HOST`     | `127.0.0.1`     | CoppeliaSim host               |
| `CSIM_PORT`     | `23001`         | ZMQ server port                |
| `VISION_SENSOR` | `/visionSensor` | Vision sensor alias            |
| `MAT_ALIAS`     | `/mat`          | Mat dummy alias                |
| `PORT`          | `8000`          | Web server port                |
| `FPS`           | `10`            | Stream FPS                     |
| `QUALITY`       | `95`            | JPEG quality (1–100)           |
| `SCALE`         | `1.25`          | Server-side upscaling          |
| `MODELS_ROOT`   | *(auto-detect)* | Override model dir (see below) |

---

## Adding Models (and Wrappers)

Place your models and small wrappers in **`suture_arm/ML_detection/`** (or point to a custom folder with `MODELS_ROOT`).

**File naming is critical (case-sensitive):**

* `mask_cuts_detector.pth` ↔ `mask_cuts_detector.py`
* `cuts_detector_best.pth`  ↔ `cuts_detector_best.py`
* `mask_cuts_detector2.pth` ↔ `mask_cuts_detector2.py`
* `cuts_detector_best2.pth` ↔ `cuts_detector_best2.py`
* `mask_cuts_detector3.pth` ↔ `mask_cuts_detector3.py`

Each wrapper must implement:

```python
def load(model_path: str):
    # return a handle (e.g., torch model + calib dict)
    ...

def predict(handle, frame_bgr):
    # return {"params":{"spacing":..., "bite":...},
    #         "cuts":[{"polyline":[[x,y],...]}, ...]}
    # Units: meters, in the 'mat' frame (z=0).
    ...
```

Optional **`calib.json`** (in the ML folder) can specify `scale_m_per_px`, `origin_px`, `y_down`, or a 3×3 homography `H` for pixel→meter mapping.

### Install models with the package (recommended)

`setup.py` already installs `ML_detection/*.pth` and `*.py` into
`share/suture_arm/ml`. If you add/remove files, rebuild:

```bash
cd ~/Documents/Dissertation/Code/dissertation/ROS/ros_ws2
rm -rf build install log
colcon build
source install/setup.bash
```

To work directly from `src/` without reinstalling, run with:

```bash
MODELS_ROOT=~/Documents/Dissertation/Code/dissertation/ROS/ros_ws2/src/suture_arm/ML_detection \
CSIM_PORT=23001 ros2 run suture_arm vision_web
```

---

## Running the Suturing Node

Start the node (it waits for a `/suture_cuts` message):

```bash
ros2 run suture_arm suturing
```

Publish a test cut (in meters, `mat` frame, z=0):

```bash
ros2 topic pub --once /suture_cuts std_msgs/String \
"data: '{\"frame_id\":\"mat\",\"cuts\":[{\"polyline\":[[-0.04,0.0],[0.04,0.0]]}],\"params\":{\"spacing\":0.006,\"bite\":0.005,\"depth\":0.003}}'"
```

Parameters:

* `spacing` – distance between stitches (default 6 mm)
* `bite` – entry↔exit offset (default 5 mm)
* `depth` – needle z-depth below the mat surface (default 3 mm)

> For now we assume `mat` is **coincident** with the robot base XY plane at z=0. If you calibrate a transform `T_base_mat`, apply it in `suture_arm_node.py`.

---

## How It Works (brief)

* **IKUR** expands UR xacro (`ur_description/urdf/ur.urdf.xacro`) for your selected robot (`ur3`) and patches `continuous`→`revolute` so **ikpy** accepts the chain. Only the 6 UR joints are active.
* **Planner** resamples the cut polyline at `spacing`, computes tangents and left/right **bites**, and generates 5-pose segments `[approach, entry, mid, exit, retract]`.
* **CoppeliaDriver** finds `/UR3` and all joint handles by name; drives **target positions** in **stepped** mode (`sim.step()` loop) for deterministic playback.

---

## Launch (optional)

You can create a combined launch to start both nodes (not CoppeliaSim itself):

```python
# suture_arm/launch/suture_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    env = {'CSIM_PORT':'23001'}  # set your ZMQ port here
    return LaunchDescription([
        Node(package='suture_arm', executable='suturing', output='screen'),
        Node(package='suture_arm', executable='vision_web', output='screen',
             env=env),
    ])
```

Run:

```bash
ros2 launch suture_arm suture_system.launch.py
```

> Start CoppeliaSim (scene + ZMQ server) separately.

---

## Troubleshooting

* **Client hangs / no stream:** ZMQ server not running or wrong port.
  In CoppeliaSim: *Add-ons → ZeroMQ remote API server* → Start (note `rpcPort`) → ▶ Play.
* **“Wrapper not found” on the web UI:**
  Filenames must match exactly (`.pth` stem ↔ `.py`). Ensure the wrappers are **installed** into `share/suture_arm/ml` or use `MODELS_ROOT=...` to point at your source folder.
* **“Could not find joint elbow\_joint”**:
  Rename `/UR3/joint` to **`elbow_joint`** to match the ROS order.
* **Object not found (mat/visionSensor)**:
  Names are case-sensitive. Verify aliases in CoppeliaSim Scene Hierarchy.
* **IK warnings (“fixed link active”):** Harmless; the driver selects the 6 moving joints explicitly.
* **Self-collision / large swings:** Reduce approach depth/spacing/bite, or start from a safe seed posture in CoppeliaSim.

---

## Useful Commands

```bash
# Check CoppeliaSim connectivity
python3 - <<'PY'
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('127.0.0.1', 23001).require('sim')
print('Version:', sim.getInt32Param(sim.intparam_program_version))
print('visionSensor alias:', sim.getObjectAlias(sim.getObject('/visionSensor')))
print('mat alias:', sim.getObjectAlias(sim.getObject('/mat')))
PY
```

---

## Repository Structure (snippet)

```
ros_ws2/
  src/
    suture_arm/
      ML_detection/
        mask_cuts_detector.pth
        mask_cuts_detector.py
        ...
      resource/ur3_suture.ttt
      suture_arm/suture_arm_node.py
      suture_arm/vision_web.py
      package.xml
      setup.py
      setup.cfg
    ur_description/
      urdf/ur.urdf.xacro
      ...
```

---

## License

TBD — add your license here.

---

## Acknowledgements

* **Universal Robots** ROS 2 description (URDF/xacro)
* **CoppeliaSim** ZeroMQ Remote API
* **ikpy** inverse kinematics library

If you run into anything weird, check `/models` in the web UI to verify the server sees your models/wrappers, and confirm the ZMQ port matches the one printed by CoppeliaSim.
