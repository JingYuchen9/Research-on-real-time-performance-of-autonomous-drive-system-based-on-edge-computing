from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_camera_sh',
            namespace='ros2_camera_sh',
            executable='ros2_camera_sh',
            name='ros2_camera_sh'
        ),
        Node(
            package='ros2_camera_strengthen',
            namespace='ros2_camera_strengthen',
            executable='ros2_camera_strengthen',
            name='ros2_camera_strengthen'
        ),
        Node(
            package='ros2_auto_drive_sh',
            executable='ros2_auto_drive_sh',
            name='ros2_auto_drive_sh',
        )
    ])