#!/usr/bin/env python3
import os, glob
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def _find_coppelia():
    # 1) If COPPELIASIM_ROOT is set and valid, use it
    root = os.environ.get('COPPELIASIM_ROOT', '')
    cand = os.path.join(root, 'coppeliaSim.sh')
    if root and os.path.isfile(cand):
        return cand

    # 2) Try common locations
    guesses = [
        '~/CoppeliaSim/coppeliaSim.sh',
        '~/Downloads/CoppeliaSim/coppeliaSim.sh',
        '~/Downloads/CoppeliaSim_Edu/coppeliaSim.sh',
    ]
    # 3) Broad wildcard (covers versioned folders, e.g. *_Ubuntu22_04)
    guesses += glob.glob(os.path.expanduser('~/Downloads/CoppeliaSim*/coppeliaSim.sh'))

    for g in guesses:
        p = os.path.expanduser(g)
        if os.path.isfile(p):
            return p

    # None found; runner will error clearly
    return ''

def generate_launch_description():
    pkg_share = get_package_share_directory('suture_arm')

    scene_arg   = DeclareLaunchArgument(
        'scene',
        default_value=PathJoinSubstitution([pkg_share, 'scenes', 'ur3_suture.ttt']),
        description='CoppeliaSim scene file (.ttt)'
    )
    headless_arg = DeclareLaunchArgument('headless', default_value='false')
    web_port_arg = DeclareLaunchArgument('web_port', default_value='8000')
    sensor_arg   = DeclareLaunchArgument('sensor', default_value='/visionSensor')
    fps_arg      = DeclareLaunchArgument('fps', default_value='100')

    # Unbuffer logs for readability
    env_unbuffered = SetEnvironmentVariable(name='PYTHONUNBUFFERED', value='1')

    # Build the command to start Coppelia via our module
    cop_path = _find_coppelia()
    cop_gui_cmd = ['python3','-m','suture_arm.coppelia_runner','--scene',LaunchConfiguration('scene')]
    cop_head_cmd = ['python3','-m','suture_arm.coppelia_runner','--scene',LaunchConfiguration('scene'),'--headless']
    if cop_path:
        cop_gui_cmd += ['--coppelia', cop_path]
        cop_head_cmd += ['--coppelia', cop_path]
    # (If not found, the runner will fall back to env var and print a clear error.)

    cop_gui = ExecuteProcess(
        cmd=cop_gui_cmd,
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    cop_headless = ExecuteProcess(
        cmd=cop_head_cmd,
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    # ROS node (OK to use Node)
    suturing_node = Node(
        package='suture_arm',
        executable='suturing',
        name='suture_arm',
        output='screen'
    )

    # Non-ROS web app -> ExecuteProcess (prevents --ros-args injection)
    vision_proc = ExecuteProcess(
        cmd=['python3','-m','suture_arm.vision_web',
             '--sensor', LaunchConfiguration('sensor'),
             '--fps',    LaunchConfiguration('fps'),
             '--port',   LaunchConfiguration('web_port')],
        output='screen'
    )

    return LaunchDescription([
        env_unbuffered,
        scene_arg, headless_arg, web_port_arg, sensor_arg, fps_arg,
        cop_gui, cop_headless,
        TimerAction(period=4.0, actions=[suturing_node]),
        TimerAction(period=4.0, actions=[vision_proc]),
    ])
