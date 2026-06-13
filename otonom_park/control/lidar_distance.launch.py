import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    rp_lidar_launch_dir = os.path.join(
        get_package_share_directory('rplidar_ros'),
        'launch'
    )
    rplidar_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(rp_lidar_launch_dir, 'view_rplidar_a2m12_launch.py')),
            launch_arguments={
                'use_sim_time': 'false',
            }.items()
        )
    
    limited_scan_node = Node(
        package='otagg_control',
        executable='limited_scan.py',
        name='limited_scan',
        parameters=[{'use_sim_time': False,}],
        output='screen'
    )

    distance_warn_node = Node(
        package='otagg_control',
        executable='distance_warn_node.py',
        name='distance_warn_node',
        parameters=[{'use_sim_time': False,}],
        output='screen'
    )



    return LaunchDescription([
        rplidar_node,
        limited_scan_node,
        distance_warn_node
    ])
