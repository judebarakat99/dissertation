# launch/vision_and_motion.launch.py
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('suture_arm')

    templates_dir = os.path.join(pkg_share, 'templates')
    models_dir    = os.path.join(pkg_share, 'ML_detection')

    sim_host = LaunchConfiguration('sim_host')
    sim_port = LaunchConfiguration('sim_port')

    return LaunchDescription([
        # ZMQ host/port used by CoppeliaSim you start manually
        DeclareLaunchArgument('sim_host', default_value='127.0.0.1', description='CoppeliaSim ZMQ host'),
        DeclareLaunchArgument('sim_port', default_value='23000',      description='CoppeliaSim ZMQ port'),

        # Let vision_web find the HTML templates and ML models
        SetEnvironmentVariable('SUTURE_ARM_TEMPLATES', templates_dir),
        SetEnvironmentVariable('SUTURE_ARM_ML',        models_dir),

        # Web / vision server (Flask) — assumes console_script "vision_web" is registered
        # If your entry point is suture_arm_node instead, change executable='vision_web' -> 'suture_arm_node'
        Node(
            package='suture_arm',
            executable='vision_web',
            name='vision_web',
            output='screen',
            parameters=[{'sim_host': sim_host, 'sim_port': sim_port}]
        ),

        # Motion executor: moves UR3 to suture ENTRY points (joint streaming via ZMQ)
        Node(
            package='suture_arm',
            executable='suture_entry_motion_node',
            name='suture_entry_motion_node',
            output='screen',
            parameters=[{'sim_host': sim_host, 'sim_port': sim_port}]
        ),
    ])
