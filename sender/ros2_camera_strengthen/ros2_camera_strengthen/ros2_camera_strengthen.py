#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import random
import signal
import sys
import os
import rclpy
import time
from collections import deque
from rclpy.node import Node
from ros2_camera_interface.msg import ImageData, ImageDataArray, ImageNoCompress
# ros2_camera_interface/ImageData[] packet_transmission
from sensor_msgs.msg import CompressedImage
import numpy as np
from cv_bridge import CvBridge
from ros2_xycar_interfaces.msg import Motor
from std_msgs.msg import UInt32MultiArray
from std_msgs.msg import UInt32
# from xycar_motor.msg import xycar_motor
# from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, Duration# ROS2 QoS类
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import threading
from pympler import asizeof


class ros2_camera_strengthen(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_camera_strengthen节点被激活")
        self.bridge = CvBridge()
        self.starttime = 0
        self.endtime = 0
        global frame_count
        frame_count = 0
        self.frame_rate = time.time()
        self.fps = time.time()
        self.file_fps = open('fps.txt', 'w')
        self.total_block_num = 0
        self.global_frame_num = 0
        self.image = None
        cb_group1 = ReentrantCallbackGroup()# ReentrantCallbackGroup()
        cb_group2 = ReentrantCallbackGroup()
        cb_group3 = ReentrantCallbackGroup()
        # self.event = threading.Event()
        # self.deal_image_thread = threading.Thread(target=self.deal_image, args=(self.event,))
        # # self.deal_image_thread = True
        # self.deal_image_thread.start()  # 启动守护线程
        self.Hight = 0
        self.Width = 0
        # self.block_size = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160] # 640 480
        self.block_size = [1, 2, 4, 5, 8, 10, 16, 20, 40, 80] # 1280 720
        self.current_size = -3  # 20
        self.buffer_queue = deque()
        # total_blocks = (1280 // self.block_size[self.current_size] * (720 // self.block_size[self.current_size]))
        # self.filter_list = list(range(1,total_blocks + 1))
        self.filter_list = []
        self.msg_buffer1 = None
        self.msg_buffer2 = None
        self.maxsize = 0
        self.rtt_list = []
        self.sync = 0
        # 打印输出图片大小
        self.file = open('picture_size.txt', 'w') 
        self.RTT_dict = dict()
        qos_profile = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            # lifespan=Duration(seconds=0.07),
            depth=0
        )

        self.camera_subscribe_ = self.create_subscription(ImageData, "ros2_xycar_camera", self.image_callback, qos_profile, callback_group=cb_group1)
        
        self.filter_subscribe_ = self.create_subscription(UInt32MultiArray, "ros2_roi_block", self.noroi_callback, qos_profile, callback_group=cb_group2)
        
        # self.loss_rate_ = self.create_subscription(UInt32, 
        #                                             "loss_rate", 
        #                                             self.loss_rate_setting, 
        #                                             1)
        
        self.command_publisher_ = self.create_publisher(ImageData, "ros2_camera_strengthen", qos_profile, callback_group=cb_group3)

        # self.timer = self.create_timer(1/30000, self.deal_image)

    def get_block_position(self, block_num, block_size, Width):
        num_blocks_per_row = Width // block_size
        row = (block_num - 1) // num_blocks_per_row
        col = (block_num - 1) % num_blocks_per_row
        x = col * block_size
        y = row * block_size
        return x, y


    # 从上一层获取视频帧数据
    def image_callback(self, msg):
        global_frame_num = msg.frame_num
        image = self.bridge.compressed_imgmsg_to_cv2(msg.image_block, "bgr8")
        self.Hight, self.Width, _ = image.shape
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]  # 6M左右
        _, encoding = cv2.imencode('.jpg', image, encode_param) 
        global_low_compress = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        # self.compress_image_h264(image, 50)
        block_num = 0

        for roi_id in self.filter_list:
            
            # 计算块的左上角坐标
            x, y = self.get_block_position(roi_id, self.block_size[self.current_size], self.Width)
            
            # 计算块的右下角坐标
            x_end = min(x + self.block_size[self.current_size], self.Width)
            y_end = min(y + self.block_size[self.current_size], self.Hight)
            
            # 从图像中提取块
            global_low_compress[y:y_end, x:x_end] = image[y:y_end, x:x_end]
        
        if len(self.filter_list) == 0:
            global_low_compress = image
######################################################       
        # for y in range(0, self.Hight, self.block_size[self.current_size]):
        #     # msg = ImageDataArray()
        #     for x in range(0, self.Weight, self.block_size[self.current_size]):
        #         block_num += 1
        #         # self.total_block_num += 1
        #         global_heigh_compress = image[y:y + self.block_size[self.current_size], x:x + self.block_size[self.current_size]]
        #         # cv2.putText(block_payload, str(block_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         if len(self.filter_list) == 0:
        #             global_low_compress = image
        #         if self.filter_list and block_num in self.filter_list:
        #             # noroi_block = self.compress_image(block_payload, 0)
        #             global_low_compress[y:y + self.block_size[self.current_size], x:x + self.block_size[self.current_size]] = global_heigh_compress

###################################################

        #         # print(self.filter_list)
        #         # if block_num not in self.filter_list:
        #         #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  
        #         #     _, encoding = cv2.imencode('.jpg', block_payload, encode_param) 
        #         #     payload = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        #         if block_num in self.filter_list:
        #             cv2.putText(block_payload, str(block_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #             payload = self.compress_image(block_payload, 70)
        #         # cv2.imshow('Image', block_payload)
        #         # cv2.waitKey(1)
        #         image_payload = self.bridge.cv2_to_compressed_imgmsg(global_low_compress, dst_format = "jpeg")
        #         send_msg = ImageData()
        #         send_msg.frame_num = global_frame_num
        #         send_msg.block_num = block_num
        #         send_msg.total_block_num = self.total_block_num
        #         send_msg.image_block = image_payload
        #         msg.packet_transmission.append(send_msg)
        #     self.command_publisher_.publish(msg)
        # image_payload = self.bridge.cv2_to_compressed_imgmsg(image, dst_format = "jpeg")
        # cv2.imshow('Image', global_low_compress)
        # cv2.waitKey(1)
        image_payload = self.bridge.cv2_to_compressed_imgmsg(global_low_compress, dst_format = "jpeg")
        msg = ImageData()
        msg.frame_num = global_frame_num
        msg.block_num = 0
        msg.total_block_num = 0
        msg.image_block = image_payload
        sizeofmsg = self.serialize_message(msg)
        self.file.write(f"{len(sizeofmsg)}\n")
        self.command_publisher_.publish(msg)
        # current_time = time.time()
        # realtime_fps = current_time - self.fps
        # print(realtime_fps)
        # self.file_fps.write(f"{realtime_fps:.2f}\n")
        # self.fps = current_time
        
        self.calcuate_frame_rate()
            # msg_size = asizeof.asizeof(msg)
            # print("帧id", image_tuple[0], "块id", block_num, "每一个包里面包含", self.block_size[self.current_size], "块")

    # 返回压缩好的图片
    def compress_image(self, image, rate):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rate]  # 压缩率10 占用带宽500kb以内，压缩率90 占用带宽最高2-3M
        _, encoding = cv2.imencode('.jpg', image, encode_param)  # 开始压缩 上面使用了JPGE压缩算法
        block_payload = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)

        return block_payload

    def loss_rate_setting(self, msg):
        setting_param = msg.data
        if setting_param:
             self.current_size = max(0, min(self.current_size + 1, 2))
        if setting_param == 0:
             self.current_size = max(0, min(self.current_size - 1, 2))

    def serialize_message(self, msg):
        from rclpy.serialization import serialize_message
        return serialize_message(msg)

    def rtt_back(self, msg):
        if msg.packet_transmission[-1].block_num == 48:
            num = msg.packet_transmission[-1].frame_num
            print(num)
            send_time = self.RTT_dict[num]
            RTT_ = (time.time() - send_time)
            print(RTT_/2)
            # self.rtt_list.append(RTT_)
            # RTT_num = len(self.rtt_list)
            # if RTT_num >= 30:
            #     ave = sum(self.rtt_list) / RTT_num
            #     print("一秒内平均RTT", ave/2)
            #     self.rtt_list.clear()
        # print(num, RTT_/2)

        # for y in range(0, 480, self.block_size):
        #     # msg = ImageDataArray()
        #     for x in range(0, 640, self.block_size):
        #         block_num += 1
        #         self.total_block_num += 1
        #         block_payload = self.image[y:y + self.block_size, x:x + self.block_size]
        #         if block_num not in self.filter_list:
        #             # 二次压缩
        #             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]  # 压缩率10 占用带宽500kb以内，压缩率90 占用带宽最高2-3M
        #             _, encoding = cv2.imencode('.jpg', block_payload, encode_param)  # 开始压缩 上面使用了JPGE压缩算法
        #             block_payload = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        #         if block_num in self.filter_list:
        #             cv2.putText(block_payload, str(block_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         image_payload = self.bridge.cv2_to_compressed_imgmsg(block_payload, dst_format = "jpeg")
        #         send_msg = ImageData()
        #         send_msg.frame_num = self.sum
        #         send_msg.block_num = block_num
        #         send_msg.total_block_num = self.total_block_num
        #         send_msg.image_block = image_payload
                # msg.packet_transmission.append(send_msg)
            # if self.msg_buffer1 is not None:
            #     self.command_publisher_.publish(self.msg_buffer1)
            # if self.msg_buffer2 is not None:
            #     self.command_publisher_.publish(self.msg_buffer2)
            # self.command_publisher_.publish(send_msg)
            # self.calcuate_frame_rate()
            # self.msg_buffer2 = self.msg_buffer1
            # self.msg_buffer1 = msg

    def noroi_callback(self, msg):
        self.filter_list = msg.data
        # upper_limit = int((1280 / self.block_size[self.current_size]) * (720 / self.block_size[self.current_size]))
        # # 初始化列表
        # result_list = list(range(1, upper_limit + 1))
        # self.filter_list = list(set(result_list) - set(self.receive_list))
        # print(self.filter_list)

    def calcuate_frame_rate(self):
        global frame_count
        frame_count += 1
        elapsed_time = time.time() - self.frame_rate
        if elapsed_time >= 1.0:  # 每隔1秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Actual FPS: {fps:.2f}")
            self.file_fps.write(f"{fps:.2f}\n")
            frame_count = 0
            self.frame_rate = time.time()

def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = ros2_camera_strengthen("ros2_camera_strengthen")  # 新建一个节点
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.file_fps.close()
    node.destroy_node()
    # rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    node.file.close()
    rclpy.shutdown()  # 关闭rclpy


