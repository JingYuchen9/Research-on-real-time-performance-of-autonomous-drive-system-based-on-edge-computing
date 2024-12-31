#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
import socket
import struct
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray


class ultrasonicPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_ultrasonic节点被激活")
        # 获取socket信息
        self.local_ip  = "192.168.123.88"
        self.local_port = 12123
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.local_ip, self.local_port))
        # 创建消息
        self.command_publisher_ = self.create_publisher(UInt32MultiArray, "ros2_ultrasonic", 10)
        # 从底层获取超声波数组
        while True:
            data, address = self.udp_socket.recvfrom(1024)
            sonic_list = struct.unpack('!{}I'.format(len(data)//4), data)
            msg = UInt32MultiArray()
            msg.data = sonic_list
            self.command_publisher_.publish(msg)
            self.get_logger().info('Published UInt32MultiArray message, message.data = %s' % msg.data)
            # 在订阅端应该使用list(msg.data)还原

def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = ultrasonicPublisher("ros2_ultrasonic")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    node.udp_socket.close()
    rclpy.shutdown()  # 关闭rclpy
