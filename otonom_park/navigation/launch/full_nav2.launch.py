import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():   

    # RViz2 configuration file
    rviz_config_file = os.path.join(get_package_share_directory('navigation_2025'), 'config', 'rviz2.rviz')
    pkg_share = get_package_share_directory('simulation_2025')
    
    # Include another launch file
    try:
        zed_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')),
            launch_arguments={'camera_model': 'zed2', 'publish_imu': 'true'}.items()
        )
        zed_launch_action = zed_launch
    except PackageNotFoundError:
        zed_launch_action = LogInfo(msg='zed_wrapper or required package zed_description not found; skipping ZED launch.')

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('rplidar_ros'), 'launch', 'rplidar_a2m12_launch.py'))
    )
    laser_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        arguments=["0", "0", "0", "0", "0", "0", "base_link", "laser"]
    )
    robot_spawner = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'robot_spawner.launch.py'))
            )
        ]
    )
    ekf_localization = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('navigation_2025'), 'launch', 'ekf_localization.launch.py'))
            )
        ]
    )

    localization = TimerAction(
        period=12.5,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('navigation_2025'), 'launch', 'localization_launch.py')),
                launch_arguments={
                    'use_sim_time': 'true'
                }.items()
            )
        ]
    )
    
    navigation = TimerAction(
        period=15.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('navigation_2025'), 'launch', 'navigation_launch.py')),
                launch_arguments={
                    'use_sim_time': 'true',
                    'map_subscribe_transient_local': 'true'
                }.items()
            )
        ]
    )
    
    rviz2 = Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', rviz_config_file],
                output='screen'
    )
   

    ld = LaunchDescription()

    ld.add_action(zed_launch_action)
    ld.add_action(lidar_launch)
    ld.add_action(laser_tf)
    ld.add_action(robot_spawner)
    ld.add_action(ekf_localization)
    ld.add_action(localization)
    ld.add_action(navigation)
    ld.add_action(rviz2)
    
    return ld
