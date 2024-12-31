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
import queue
from rclpy.node import Node
from ros2_camera_interface.msg import ImageData, ImageDataArray, ImageNoCompress
# ros2_camera_interface/ImageData[] packet_transmission
import numpy as np
from cv_bridge import CvBridge
# from ros2_xycar_interfaces.msg import Motor
from std_msgs.msg import UInt32MultiArray
from std_msgs.msg import UInt32
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sortedcontainers import SortedDict
# from xycar_motor.msg import xycar_motor
# from sensor_msgs.msg import Image
from collections import deque
import socket
###################### ccnet #############
current_directory = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path 中
if current_directory not in sys.path:
    sys.path.append(current_directory)
import CCNet as Network
import torch
import torchvision.transforms as transforms

def signal_handler(sig, frame):
    import time
    time.sleep(2)
    # os.system('killall -9 python rosout')
    sys.exit(0)

class auto_driveSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_auto_drive_sh1节点被激活")
        cb_group1 = MutuallyExclusiveCallbackGroup()# ReentrantCallbackGroup()
        cb_group2 = MutuallyExclusiveCallbackGroup()
        self.bridge = CvBridge() 
        self.buffer_queue = deque()
        self.image480p = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        self.image720p = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
        self.dataSize_Heigh = 0
        self.dataSize_Weight = 0
        self.image = None
        self.ImageData = None
        self.block_size = 0
        self.last_time = time.time()
        self.ave_fps = []
        self.ave_diff = []
        # self.target_ip = "192.168.31.152"
        # self.target_port = 12129
        # self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.global_msg = None
        self.start_time = 0
         ########### load setnetwork ##############
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.model = Network.SegNetwork().to(self.device)
        self.model = Network.LETNet().to(self.device)
        self.model.load_state_dict(torch.load("/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/pkl/letnet.pkl", weights_only=True))
        # self.model.load_state_dict(torch.load("/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/pkl/baseline.pth", weights_only=True))
        self.model.eval()
        self.seg_starttime = time.time()
        # self.model_fps = open('picture_size.txt', 'w')

        qos_profile = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        # self.camera_subscribe_  = self.create_subscription(ImageData,
        #                                                     "ros2_camera_strengthen", 
        #                                                     self.segmentation_callback, 
        #                                                     qos_profile, callback_group=cb_group1)
        
        self.camera_subscribe_  = self.create_subscription(ImageNoCompress,
                                                            "ros2_camera_strengthen", 
                                                            self.auto_drive_callback, 
                                                            qos_profile, callback_group=cb_group2)
        
        self.ImageData_publish = self.create_publisher(ImageData, "ros2_auto_drive", qos_profile)

        self.seg_result = self.create_publisher(ImageData, "seg_result", 10)

        # # 是否开启模型
        # self.player_timer = self.create_timer(1/60, self.segmentation_callback, callback_group=cb_group1)
        
        self.filter_publish_ = self.create_publisher(UInt32MultiArray, "ros2_noroi_block", 10)

        # self.player_timer = self.create_timer(1/80000, self.deal_image, callback_group=cb_group1)
        # self.player_timer = self.create_timer(1/80000, self.deal_image)
    def auto_drive_callback(self, msg):
        self.global_msg = msg
        self.ImageData_publish.publish(msg)
        # current_time = time.time()
        # fps = 1/ (current_time - self.last_time)
        # self.last_time = current_time
        # self.ave_fps.append(fps)
        # print("ave_fps =", sum(self.ave_fps) / len(self.ave_fps))

    def segmentation_callback(self):
        
        if self.global_msg is None:
            return
        gen_frame = self.bridge.compressed_imgmsg_to_cv2(self.global_msg.image_block, "bgr8")
        # resized_image = self.resize_image(gen_frame, 50)
        result = self.use_model(gen_frame)
        msg = ImageData()
        msg.frame_num = self.global_msg.frame_num
        msg.block_num = 0
        msg.total_block_num = 0
        msg.image_block = self.bridge.cv2_to_compressed_imgmsg(result, dst_format = "jpeg")
        self.seg_result.publish(msg)
        current_time = time.time()
        diff = 1 / (current_time - self.seg_starttime)
        print("diff =", diff)
        self.ave_diff.append(diff)
        self.seg_starttime = current_time
        print("ave_diff =", sum(self.ave_diff) / len(self.ave_diff))

    def color_adjust(self, img):
        B,G,R = cv2.split(img)
        img=cv2.merge([B,G,R])
        return img

    def get_array_inverse(self, arr1):
        return np.where(arr1 == 0, 1, 0)

    def seg_mask_im_macro(self, arr2, noroi_block_size=20):
        arr1 = np.array(arr2)
        h, w = arr1.shape
        block_id = 0
        noroi_list = []
        for i in range(0, h, noroi_block_size):
            for j in range(0, w, noroi_block_size):
                block_id += 1
                macro_block = arr1[i:i+noroi_block_size, j:j+noroi_block_size]
                if np.any(macro_block==0):
                    noroi_list.append(block_id)
                    arr1[i:i+noroi_block_size, j:j+noroi_block_size] = 0
        return arr1, noroi_list


    def resize_image(self, image, scale_percent):

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # Resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized


    def use_model(self, frame):

        # 应用预处理
        input_tensor = self.preprocess(frame).to(self.device)

        # 增加 batch 维度
        input_batch = input_tensor.unsqueeze(0)  # 形状变为 (1, 3, H, W)

        with torch.no_grad():
            segresult = self.model(input_batch)

        segresulti = segresult[0,0,:,:,]*0+segresult[0,1,:,:,]*1+segresult[0,2,:,:,]*2+segresult[0,3,:,:,]*3
        # io.imsave('./seg'+"haha"+'.png', (segresulti.data.cpu().numpy() * 255).astype(np.uint8))
        # 转换为numpy数组

        segresulti_np = segresulti.data.cpu().numpy()

        # 将segresulti缩放到0-255并转换为uint8类型
        segresulti_np = (segresulti_np * 255).astype(np.uint8)

        # image = self.overlay_grayscale_on_color(frame, segresulti_np)
        # 转换二值图,黑变白,白变黑
        arr_im = self.get_array_inverse(segresulti_np)
        # 映射到20x20宏块,由于cv2.resize 使用的是插值法进行缩放,因此缩小前掩码块(20x20)的位置和缩小后的掩码块(40x40)位置是一一对应的
        seg_mask_im_macro_old, macro_block_ids = self.seg_mask_im_macro(arr_im, noroi_block_size=40)
        noroi_block_msg = UInt32MultiArray()
        noroi_block_msg.data = macro_block_ids
        self.filter_publish_.publish(noroi_block_msg)
        img = None
        # 下面的函数用于可视化
        im = np.array(seg_mask_im_macro_old)
        seg_mask_uim_macro = np.expand_dims(im,2).repeat(3,axis=2)
        im_images = seg_mask_uim_macro * frame
        img = self.color_adjust(im_images)
        if img is None:
            return
        if img is not None:
            return img.astype(np.uint8)
    
    def overlay_grayscale_on_color(self, color_frame, grayscale_frame):
        # 确保彩色图像和灰度图像的尺寸相同
        if color_frame.shape[:2] != grayscale_frame.shape:
            raise ValueError("彩色图像和灰度图像的尺寸必须相同")

        # 将灰度图像转换为三通道
        grayscale_3ch = cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2BGR)

        # 创建一个Alpha通道，白色部分为0(透明)，黑色部分为255(不透明)
        alpha_channel = cv2.bitwise_not(grayscale_frame)  # 反转灰度图，使黑色为255，白色为0

        # alpha_channel = np.zeros_like(alpha_channel)
        # alpha_channel[alpha_channel == 255] = 0
        # alpha_channel[alpha_channel == 0] = 255

        # 合并灰度图像和Alpha通道
        grayscale_4ch = cv2.merge((grayscale_3ch, alpha_channel))

        # 将彩色图像转换为BGRA
        color_frame_4ch = cv2.cvtColor(color_frame, cv2.COLOR_BGR2BGRA)

        # 叠加灰度图到彩色图像上
        result_frame = cv2.addWeighted(color_frame_4ch, 1, grayscale_4ch, 1, 0)

        return result_frame
    
    def deal_image(self):
        if len(self.buffer_queue) <= 0 :# (self.dataSize * 3):
            return 
        data = self.buffer_queue.popleft()
        print("当前缓存大小", len(self.buffer_queue))
        self.ImageData_publish.publish(data)

def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init(args=args)  # 初始化rclpy
    node = auto_driveSubscriber("ros2_audo_drive_sh1")  # 新建一个节点
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    # rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown()  # 关闭rclpy


