#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('suture_arm')

    # --- Launch args (no coppelia path arg!) ---
    scene_arg   = DeclareLaunchArgument(
        'scene',
        default_value=PathJoinSubstitution([pkg_share, 'scenes', 'ur3_suture.ttt']),
        description='CoppeliaSim scene file (.ttt)'
    )
    headless_arg = DeclareLaunchArgument('headless', default_value='false')
    web_port_arg = DeclareLaunchArgument('web_port', default_value='8000')
    sensor_arg   = DeclareLaunchArgument('sensor', default_value='/visionSensor')
    fps_arg      = DeclareLaunchArgument('fps', default_value='10')

    # Helpful for real-time logs (no buffering)
    env_unbuffered = SetEnvironmentVariable(name='PYTHONUNBUFFERED', value='1')

    # --- Start CoppeliaSim via our module; it will auto-detect COPPELIASIM_ROOT ---
    cop_gui = ExecuteProcess(
        cmd=[
            'python3', '-m', 'suture_arm.coppelia_runner',
            '--scene', LaunchConfiguration('scene'),
        ],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    cop_headless = ExecuteProcess(
        cmd=[
            'python3', '-m', 'suture_arm.coppelia_runner',
            '--scene', LaunchConfiguration('scene'),
            '--headless',
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    # --- Your nodes ---
    suturing_node = Node(
        package='suture_arm',
        executable='suturing',
        name='suture_arm',
        output='screen'
    )
    vision_node = Node(
        package='suture_arm',
        executable='vision_web',
        name='vision_web',
        output='screen',
        arguments=[
            '--sensor', LaunchConfiguration('sensor'),
            '--fps', LaunchConfiguration('fps'),
            '--port', LaunchConfiguration('web_port'),
        ]
    )

    # Give the simulator a moment to bring up ZMQ
    delayed_suturing = TimerAction(period=4.0, actions=[suturing_node])
    delayed_vision   = TimerAction(period=4.0, actions=[vision_node])

    return LaunchDescription([
        env_unbuffered,
        scene_arg, headless_arg, web_port_arg, sensor_arg, fps_arg,
        cop_gui, cop_headless,
        delayed_suturing, delayed_vision,
    ])
