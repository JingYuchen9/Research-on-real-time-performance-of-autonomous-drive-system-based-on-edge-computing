#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# ros2_xycar_interfaces/msg/XycarMotor
from ros2_xycar_interfaces.msg import Motor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类
from rclpy.duration import Duration
import socket
import numpy as np


class PublishToRos2RunMotor(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("节点已启动：%s!" % name)
        self.target_ip  = "192.168.123.89"
        self.target_port = 12125
        qos_profile = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )
        self.command_subscriber_ = self.create_subscription(Motor, "XycarMotor", self.process_motordata, qos_profile)
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.target_ip, self.target_port))

    def process_motordata(self, msg):
        motor_angle = msg.angle
        motor_speed = msg.speed

        # 走TCP传输数据给底层
        motor_control = np.array([motor_angle, motor_speed], dtype=np.int32)
        print("发送参数", motor_control)
        tcp_packet = motor_control.tobytes()
        self.tcp_socket.send(tcp_packet)

    def close_connection(self):
        try:
            motor_control = np.array([90, 90], dtype=np.int32)
            tcp_packet = motor_control.tobytes()
            self.tcp_socket.send(tcp_packet)
        except socket.error as e:
            self.get_logger().error("发送关闭数据失败: %s" % str(e))
        finally:
            self.tcp_socket.close()
            self.get_logger().info("TCP连接已断开")

def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = PublishToRos2RunMotor("ros2_xycar_motor")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C)``
    node.tcp_socket.close()
    rclpy.shutdown()  # 关闭rclpy
