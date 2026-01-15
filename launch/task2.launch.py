import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # --- ARGUMENTS ---
    # This allows you to say "target_color:='red'" in the terminal
    color_arg = DeclareLaunchArgument(
        'target_color',
        default_value='green',
        description='Which gate to hunt: "red" or "green"'
    )

    # --- NODE 1: CAMERA (Terminal 1) ---
    # We include this so you don't need a separate terminal for the camera
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera_driver',
        parameters=[{
            'camera_info_url': 'file:///home/yashas/srmauv_ws/src/aquavision_task2_pkg/config/default_cam.yaml'
        }],
        remappings=[
            ('/image_raw', '/camera/image_raw')
        ]
    )

    # --- NODE 2: VISION PROCESSOR (Terminal 2 + 3) ---
    # Starts the node AND sets the parameter immediately
    vision_node = Node(
        package='aquavision_task2_pkg',
        executable='vision_processor', # Matches your 'ros2 run' command
        name='vision_processor',
        output='screen',
        parameters=[{
            'target_color': LaunchConfiguration('target_color')
        }]
    )

    return LaunchDescription([
        color_arg,
        camera_node,
        vision_node
    ])