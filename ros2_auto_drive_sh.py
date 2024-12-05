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
from rclpy.node import Node
from ros2_camera_interface.msg import ImageData, ImageDataArray, ImageNoCompress
import numpy as np
from cv_bridge import CvBridge
from ros2_xycar_interfaces.msg import Motor 
from std_msgs.msg import UInt32MultiArray
from std_msgs.msg import UInt32
from sensor_msgs.msg import CompressedImage
# from xycar_motor.msg import xycar_motor
# from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from collections import defaultdict
import threading
import socket
###################### HybridNets #############
import torch
from torch.backends import cudnn
sys.path.append('/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh')
from backbone import HybridNetsBackbone
# from backbone_update import HybridNetsBackbone
from glob import glob
import sys
sys.path.append('/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/utils')
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from torchvision import transforms
import argparse
from utils.constants import *
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
# sys.path.append('/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/projects')
parser.add_argument('--project', type=str, default='bdd100k', help='Project file that contains parameters')
# mobilenetv3_large_100_miil_in21k
parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                        'https://github.com/rwightman/pytorch-image-models')
parser.add_argument('--pretrain_weight', type=str, default='')
parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
# parser.add_argument('--source', type=str, default='/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/demo/video/1.mp4', help='The demo video folder')
# parser.add_argument('--source', type=int, default=0, help='The demo video folder')
# parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
# parser.add_argument('--load_weights', type=str, default='/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/weights/hybridnets_msca.pth')
parser.add_argument('--load_weights', type=str, default='/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/weights/hybridnets.pth')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=False, help="Use float16 for faster inference")
args = parser.parse_args()

def signal_handler(sig, frame):
    import time
    time.sleep(2)
    # os.system('killall -9 python rosout')
    sys.exit(0)

class auto_driveSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_auto_drive_sh节点被激活")
        self.bridge = CvBridge()
        # socket设置
        self.target_ip = "192.168.31.152"
        # self.target_ip = "10.30.115.50"
        self.target_port = 12129
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        global Offset
        global Gap
        global time_rate
        # 图像拼接
        global x
        global y
        self.start_time = time.time()
        self.frame = dict()
        # 数据分片大小
        self.block_size = 80
        global frame_count
        frame_count = 0
        self.frame_rate = time.time()
        # self.thread = threading.Thread(target=self.subscription_callback)
        # self.lock = threading.Lock()
        # self.thread.start()
        self.current_framenum = 0
        self.weight_dict = dict()
        self.image480p = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        self.image720p = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
        # self.image = None
        # self.dataSize_Heigh = 0
        # self.dataSize_Weight = 0
        # self.pointer_frame = 1
        # self.buffer_pool = defaultdict(list)
        self.global_image = np.empty(shape=[0])
        x = 0
        y = 0 
        time_rate = 0
        # image = np.empty(shape=[0])
        self.height = 0
        self.width = 0
        Offset = 200
        Gap = 160
        self.last_time = time.time()
        self.segment_last_time = time.time()
        self.ave_fps = []
        self.seg_ave_fps = []
        self.ave_w = []
        self.N = 1.4 # 你可以调整这个值
        self.angle_history = []
        self.lpos = 0
        self.rpos = 0
        self.previous_angle = 90
        ########## 之后会频繁调整速度 ##############
        
        self.speed = 120
        self.start_time = time.time()
        self.global_msg = None

        self.params = Params(f'/home/jing/ros2_ws/src/ros2_auto_drive_sh/ros2_auto_drive_sh/projects/{args.project}.yml')
        self.FocalLength = [550.2377150910099,5.470046088166852e+02] # 相机标定焦距
        self.color_list_seg = {}
        for seg_class in self.params.seg_list:
            # edit your color here if you wanna fix to your liking
            self.color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
        compound_coef = args.compound_coef
        # source = args.source
        weight = args.load_weights
        self.model = None
        anchors_ratios = self.params.anchors_ratios
        anchors_scales = self.params.anchors_scales

        self.threshold = args.conf_thresh
        self.iou_threshold = args.iou_thresh

        self.use_cuda = args.cuda
        self.use_float16 = args.float16
        cudnn.fastest = True
        cudnn.benchmark = True

        self.obj_list = self.params.obj_list
        self.seg_list = self.params.seg_list

        self.color_list = standard_to_bgr(STANDARD_COLORS)
        self.resized_shape = self.params.model['image_size']
        if isinstance(self.resized_shape, list):
            self.resized_shape = max(self.resized_shape)
        normalize = transforms.Normalize(
            mean=self.params.mean, std=self.params.std
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        # print(x.shape)
        weight = torch.load(weight, map_location='cuda' if self.use_cuda else 'cpu', weights_only=False)
        weight_last_layer_seg = weight.get('model', weight)['segmentation_head.0.weight']
        if weight_last_layer_seg.size(0) == 1:
            self.seg_mode = BINARY_MODE
        else:
            if self.params.seg_multilabel:
                self.seg_mode = MULTILABEL_MODE
                print("Sorry, we do not support multilabel video inference yet.")
                print("In image inference, we can give each class their own image.")
                print("But a video for each class is meaningless.")
                print("https://github.com/datvuthanh/HybridNets/issues/20")
                exit(0)
            else:
                self.seg_mode = MULTICLASS_MODE
        print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", self.seg_mode)
        if args.backbone == 'mobilenetv3_large_100_miil_in21k':
            self.model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(self.obj_list), ratios=eval(anchors_ratios),
                                    scales=eval(anchors_scales), seg_classes=len(self.seg_list), backbone_name=args.backbone,
                                    seg_mode=self.seg_mode, seg_p2_in_channels=24, pretrain_weight=args.pretrain_weight)
        if args.backbone is None:
            self.model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(self.obj_list), ratios=eval(anchors_ratios),
                                    scales=eval(anchors_scales), seg_classes=len(self.seg_list), backbone_name=args.backbone,
                                    seg_mode=self.seg_mode)
        self.model.load_state_dict(weight.get('model', weight))

        self.model.requires_grad_(False)
        self.model.eval()
        ###############################################
        cb_group1 = ReentrantCallbackGroup()# ReentrantCallbackGroup()
        cb_group2 = ReentrantCallbackGroup()
        cb_group3 = ReentrantCallbackGroup()

        if self.use_cuda:
            self.model = self.model.cuda()
            if self.use_float16:
                self.model = self.model.half()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        qos_profile1 = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )
        
        qos_profile2 = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )
        
        qos_profile3 = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )
        
        # 自动驾驶节点
        self.camera_subscribe_  = self.create_subscription(ImageData, "ros2_auto_drive", self.auto_drive_callback, qos_profile1, callback_group=cb_group1)
        # self.camera_subscribe_  = self.create_subscription(ImageData, "ros2_camera_strengthen", self.auto_drive_callback, qos_profile)
        
        # 分割节点可视化
        self.seg_result  = self.create_subscription(ImageData, "seg_result", self.seg_result, qos_profile1)
        
        # 电机节点
        self.motor_publish_ = self.create_publisher(Motor, "XycarMotor", qos_profile2, callback_group=cb_group2)

        # 返回RTT
        self.rtt_back = self.create_publisher(UInt32, "rtt", qos_profile3, callback_group=cb_group3)
        
        # 视频切片组合节点,暂时用不到
        # self.timer = self.create_timer(1/20000, self.get_StreamFromPool)

        # 开始车道线识别
        # self.timer = self.create_timer(1/30, self.start)

    def seg_result(self, msg):
        frame_id = msg.frame_num
        image = self.bridge.compressed_imgmsg_to_cv2(msg.image_block, "bgr8")
        current_time = time.time()
        fps = 1/ (current_time - self.segment_last_time)
        self.segment_last_time = current_time
        self.seg_ave_fps.append(fps)
        print("seg_ave_fps =", sum(self.seg_ave_fps) / len(self.seg_ave_fps))
        # self.udp_socket.sendto(frame_id.to_bytes((frame_id.bit_length() + 7) // 8, byteorder='big'), (self.target_ip, self.target_port))
        cv2.imshow('seg_result', image)
        cv2.waitKey(1)

    def auto_drive_callback(self, msg):
        self.global_msg = msg
        # frame_id = msg.frame_num
        # self.global_image = self.bridge.compressed_imgmsg_to_cv2(msg.image_block, "bgr8")
        # # image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8") 
        # # gen_frame = self.merge_image(image)
        self.height = 720
        self.width = 1280
        # self.height, self.width, channels = self.global_image.shape
        # current_time = time.time()
        # fps = 1/ (current_time - self.last_time)
        # self.last_time = current_time
        # self.ave_fps.append(fps)
        # print("ave_fps =", sum(self.ave_fps) / len(self.ave_fps))
        # self.udp_socket.sendto(frame_id.to_bytes((frame_id.bit_length() + 7) // 8, byteorder='big'), (self.target_ip, self.target_port))
        # cv2.imshow('Image', self.global_image)
        # cv2.waitKey(1)
        self.start()
        del msg
        
    def merge_image(self, frame):

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

        return segresulti_np
    
    def Stitching_image(self, data):
        y_offset = (int(data[-1].block_num / len(data)) - 1) * self.block_size
        x_offset = 0
        for imageData in data:
            image_part = self.bridge.compressed_imgmsg_to_cv2(imageData.image_block, "bgr8")
            self.image[y_offset:y_offset + self.block_size, 
                x_offset:x_offset + self.block_size] = image_part
            x_offset += self.block_size
        return self.image
    
    def img_callback(self, msg):
        data = msg.packet_transmission
        # 自动配置 720p or 480p
        if self.image is None or self.dataSize_Heigh == 0 or self.dataSize_Weight == 0:
            if len(data) == 8:
                self.image = self.image480p
                self.dataSize_Heigh = 6
                self.dataSize_Weight = 8
            if len(data) == 16:
                self.image = self.image720p
                self.dataSize_Heigh = 9
                self.dataSize_Weight = 16

        self.buffer_pool[data[-1].frame_num].append(data)

    # def get_StreamFromPool(self):
    #     if len(self.buffer_pool) < 2:
    #         return
    #     value = self.buffer_pool.pop(self.pointer_frame, None)
    #     if value is None or len(value) < self.dataSize_Heigh - 1:
    #         self.pointer_frame += 1
    #         return
    #     for data in value:
    #         if data[-1].block_num == (self.dataSize_Heigh*self.dataSize_Weight):
    #             gen_frame = self.Stitching_image(data)
    #             gen_frame = self.merge_image(gen_frame)
    #             current_time = time.time()
    #             fps = 1/ (current_time - self.last_time)
    #             self.last_time = current_time
    #             self.ave_fps.append(fps)
    #             print("ave_fps =", sum(self.ave_fps) / len(self.ave_fps))
    #             cv2.imshow('Image', gen_frame)
    #             cv2.waitKey(1)
    #             # self.calcuate_frame_rate()
    #             self.udp_socket.sendto(data[-1].frame_num.to_bytes((data[-1].frame_num.bit_length() + 7) // 8, byteorder='big'), (self.target_ip, self.target_port))            
                
    #         if data[-1].block_num < (self.dataSize_Heigh*self.dataSize_Weight):
    #             _ = self.Stitching_image(data)
    #     self.pointer_frame += 1
        
    def calcuate_frame_rate(self):
        global frame_count
        frame_count += 1
        elapsed_time = time.time() - self.frame_rate
        if elapsed_time >= 1.0:  # 每隔1秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Actual FPS: {fps:.2f}")
            frame_count = 0
            self.frame_rate = time.time()

    # publish xycar_motor msg
    def drive(self, Angle, Speed):
        # Angle = max(-50, min(Angle, 50))
        # # Angle = int((Angle * 1.4) + 90)
        # Angle = int((Angle * self.N) + 90)
        # print("Angle", Angle)
        # print("Speed", Speed)
        msg = Motor()
        msg.speed = Speed
        msg.angle = Angle
        self.motor_publish_.publish(msg)

    def limit_angle_change(self, angle, previous_angle, max_change=10):
        # 计算与前一次转角的差值
        delta_angle = angle - previous_angle
        
        # 如果差值超过最大限制，则限制到最大限制
        if delta_angle > max_change:
            angle = previous_angle + max_change
        elif delta_angle < -max_change:
            angle = previous_angle - max_change

        return angle

    # def lane_right_fitting(self, contour, frame):

    #     x_ = contour[:, 0, 0]
    #     y_ = contour[:, 0, 1]
    #     # 使用最小二乘法进行直线拟合，返回拟合结果的系数
    #     coefficients = np.polyfit(x_, y_, 1)

    #     slope, intercept = coefficients

    #     # 扩展直线的长度，这里假设为画面的边界
    #     x1 = 0
    #     y1 = int(slope * x1 + intercept)

    #     x2 = frame.shape[1] - 1
    #     y2 = int(slope * x2 + intercept)

    #     # 绘制原始点和拟合直线
    #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # 绘制黄色直线，线宽为4
    #     return frame ,slope, intercept
    
    # def lane_left_fitting(self, contour, frame):
    #     x_ = contour[:, 0, 0]
    #     y_ = contour[:, 0, 1]
    #     # 使用最小二乘法进行直线拟合，返回拟合结果的系数
    #     coefficients = np.polyfit(x_, y_, 1)

    #     slope, intercept = coefficients

    #     # 扩展直线的长度，这里假设为画面的边界
    #     x1 = 0
    #     y1 = int(slope * x1 + intercept)

    #     x2 = frame.shape[1] - 1
    #     y2 = int(slope * x2 + intercept)

    #     # 绘制原始点和拟合直线
    #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # 绘制黄色直线，线宽为4
    #     return frame ,slope, intercept

###################### 第三次尝试 kmeans #################

    def get_line_params(self, lines):
        n_clusters=2
        # 存储每条线段的斜率和截距
        slopes_intercepts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                # 垂直线，斜率设为无穷大
                m = float('inf')
                b = x1  # 对于垂直线，b通常不适用，但这里可以用x1代替
            else:
                m = float(y2 - y1) / float(x2 - x1)
                b = y1 - m * x1

            slopes_intercepts.append([m, b])

        # 确保有足够的样本进行聚类
        n_samples = len(slopes_intercepts)

        if n_samples == 0:
            return 0, 0
        
        if n_samples < n_clusters:
            avg_m = np.mean([item[0] for item in slopes_intercepts])
            avg_b = np.mean([item[1] for item in slopes_intercepts])
            print(f"样本不足 {n_clusters} 个，返回平均值：m = {avg_m}, b = {avg_b}")
            return avg_m, avg_b
        
        # 将列表转换为numpy数组
        data = np.array(slopes_intercepts)

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        labels = kmeans.labels_

        # 计算每一类的平均斜率和截距
        class_params = {i: {'m': [], 'b': []} for i in range(n_clusters)}

        for i, (m, b) in enumerate(slopes_intercepts):
            label = labels[i]
            class_params[label]['m'].append(m)
            class_params[label]['b'].append(b)

        avg_slopes = []
        avg_intercepts = []

        for i in range(n_clusters):
            if len(class_params[i]['m']) > 0:
                avg_m = np.mean(class_params[i]['m'])
                avg_b = np.mean(class_params[i]['b'])
                avg_slopes.append(avg_m)
                avg_intercepts.append(avg_b)

        # 计算绝对值最大的斜率和截距
        # 修改
        max_slope = max(avg_slopes, key=abs) if avg_slopes else 0
        max_intercept = max(avg_intercepts, key=abs) if avg_intercepts else 0

        return max_slope, max_intercept
    

    def get_line_pos(self, img, Width ,Height, lines, left=False, right=False):
        global Offset, Gap
        m, b = self.get_line_params(lines)
        if m == 0 and b == 0:
            if left:
                pos = 0
            if right:
                pos = Width
        else:
            y = Gap / 2
            pos = (y - b) / m

            b += Offset
            x1 = (Height - b) / float(m)
            x2 = ((Height/2) - b) / float(m)

            cv2.line(img, (int(x1), int(Height)), (int(x2), int((Height/2))), (255, 0,0), 3, cv2.LINE_AA)

        return img, int(pos), m

    def divide_left_right(self, lines, Width):
        low_slope_threshold = 0.025
        high_slope_threshold = 10

        # calculate slope & filtering with threshold
        slopes = []
        new_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]


            if x2 - x1 == 0:
                slope = 0
            else:
                slope = float(y2-y1) / float(x2-x1)
            
            if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
                slopes.append(slope)
                new_lines.append(line[0])

        # divide lines left to right
        left_lines = []
        right_lines = []

        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]

            x1, y1, x2, y2 = Line

            if (slope < 0) and (x2 < Width/2 - 90):
                left_lines.append([Line.tolist()])
            elif (slope > 0) and (x1 > Width/2 + 90):
                right_lines.append([Line.tolist()])

        return left_lines, right_lines
    
    def draw_lines(self, img, lines):
        global Offset
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.line(img, (x1, y1+ Offset), (x2, y2 + Offset), color, 2)
        return img
    
    def draw_rectangle(self, img, lpos, rpos, offset=0):
        center = int((lpos + rpos) / 2)

        cv2.rectangle(img, (lpos - 5, 15 + offset),
                        (lpos + 5, 25 + offset),
                        (0, 255, 0), 2)
        cv2.rectangle(img, (rpos - 5, 15 + offset),
                        (rpos + 5, 25 + offset),
                        (0, 255, 0), 2)
        cv2.rectangle(img, (int(center - 5), 15 + offset),
                    (int(center + 5), 25 + offset),
                    (0, 255, 0), 2)    
        cv2.rectangle(img, (315, 15 + offset),
                        (325, 25 + offset),
                        (0, 0, 255), 2)
        return img

    # 计算障碍物和视频下方中心的斜率用于避障
    def calculate_slope(self, x1, y1, x2, y2):
        if x2 == x1:  # 避免除以零
            return float('inf')  # 无穷大
        return (y2 - y1) / (x2 - x1)
    
    def calculate_angle(self, slope):
        # 计算斜率对应的角度（弧度）
        # 将弧度转换为角度
        return abs(90 - math.degrees(math.atan(slope)))
    
    # show image and return lpos, rpos
    def process_image(self, frame, Height, Width, input_img):
        global Offset, Gap
        global time_rate

        # HoughLinesP
        roi = frame[Offset : Offset+Gap, 0:Width]
        # cv2.imshow('roi', roi)
        # cv2.waitKey(1)
        all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,10,5)
        cvmodel = False

        # divide left, right lines
        if all_lines is None:
            if Width == 640:
                return 0, 640
            if Width == 1280:
                return 1280
        left_lines, right_lines = self.divide_left_right(all_lines, Width)
        # get center of lines
        frame, lpos, left_slope = self.get_line_pos(frame, Width, Height, left_lines, left=True)
        frame, rpos, right_slope = self.get_line_pos(frame, Width, Height, right_lines, right=True)

         # 如果没有检测到车道线
        if left_slope == 0 or right_slope == 0:
            print("车道线丢失")
            self.N  = 1.4
        
        # 计算两条车道线的平均斜率
        # avg_slope = abs((left_slope + right_slope) / 2)
        # print(avg_slope)
        print(left_slope, right_slope)

        # 如果平均斜率接近于零，认为是直线
        if all(0.5 < abs(slope) < 1 for slope in (left_slope, right_slope)):
            self.N = 0.5
            print("直线")
        else:
            # 否则为弯道
            self.N  = 1.4
            print("弯道")

        # draw lines
        frame = self.draw_lines(frame, left_lines)
        frame = self.draw_lines(frame, right_lines)

        # draw rectangle
        frame = self.draw_rectangle(frame, lpos, rpos, offset=Offset)
        # cv2.imshow('calibration', frame)
        # cv2.waitKey(1)
        
        return lpos, rpos


    def start(self):

        global Offset, Gap

        if self.height not in [720, 480]:
            return
        frame_id = self.global_msg.frame_num
        self.global_image = self.bridge.compressed_imgmsg_to_cv2(self.global_msg.image_block, "bgr8")
        frame = cv2.cvtColor(self.global_image, cv2.COLOR_BGR2RGB)

        r = self.resized_shape / max(self.height, self.width)  # resize image to img_size
        input_img = cv2.resize(frame, (int(self.width * r), int(self.height * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]
        (input_img, _), ratio, pad = letterbox((input_img, None), auto=False,
                                                    scaleup=True)

        shapes = ((self.height, self.width), ((h / self.height, w / self.width), pad))

        if self.use_cuda:
            x = self.transform(input_img).cuda()
        else:
            x = self.transform(input_img)

        x = x.to(torch.float16 if self.use_cuda and self.use_float16 else torch.float32)
        x.unsqueeze_(0)
        with torch.no_grad():
            features, regression, classification, anchors, seg = self.model(x)
            seg = seg[:, :, int(pad[1]):int(h+pad[1]), int(pad[0]):int(w+pad[0])]
            # (1, C, W, H) -> (1, W, H)
            if self.seg_mode == BINARY_MODE:
                seg_mask = torch.where(seg >= 0, 1, 0)
                seg_mask.squeeze_(1)
                print("binary mode")
            else:
                _, seg_mask = torch.max(seg, 1)
            # (1, W, H) -> (W, H)
            seg_mask_ = seg_mask[0].squeeze().cpu().numpy()
            # 调整回原来的大小
            # seg_mask_ = cv2.resize(seg_mask_, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
            # 颜色相关调整
            # color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            # for index, seg_class in enumerate(self.params.seg_list):
            #     # print(index, seg_class)
            #     if index == 1:
            #         color_seg[seg_mask_ == index+1] = self.color_list_seg[seg_class]
            # color_seg = color_seg[..., ::-1]  # RGB -> BGR
            # # cv2.imwrite('seg_only_{}.jpg'.format(i), color_seg)
            # color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
            # frame[color_mask != 0] = frame[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            # frame = frame.astype(np.uint8)

            lane_index = 1  # 假设车道类别的索引是1
            lane_mask = np.zeros_like(seg_mask_)
            lane_mask[seg_mask_ == lane_index + 1] = 255  # +1是因为seg_mask_中的类别索引从1开始
            # 分割出车道线 
            # cv2.imshow("jing", lane_mask.astype(np.uint8))
            # cv2.waitKey(1)
            
            self.lpos, self.rpos = self.process_image(lane_mask.astype(np.uint8), h, w, input_img)

            center = (self.lpos + self.rpos) / 2
            # print(diff)
            # if diff <= 5:
            #     self.N = 0.8
            #     print("直线")
            # else:
            #     self.N = 1.4
            #     print("弯道")
            angle = int(-(w/2 - center))
            # self.drive(angle, self.speed)
            # contours, _ = cv2.findContours(lane_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 获取左右两条车道线的直线
            # frame, m2, b2 = self.lane_right_fitting(contours[-1], frame)
            # current_time = time.time()
            # print(1 / (current_time - self.start_time))
            # self.start_time = current_time
            # draw_line(frame, m2, b2, 2)
            # frame, rpos = self.get_line_pos(frame, m2, b2, right=True)

            # frame, m1, b1 = self.lane_left_fitting(contours[-2], frame)
            # # draw_line(frame, m2, b2, 2)
            # frame, lpos = self.get_line_pos(frame, m1, b1, left=True)
            # # 计算两条车道线交点坐标
            # x_intersect = (b2 - b1) / (m1 - m2)
            # y_intersect = m1 * x_intersect + b1 
            # x1 = (self.height - b1) / m1
            # x2 = (self.height - b2) / m2
            # # 标记两条车道线交点与视频底部组成的三角形的三个点
            # cv2.circle(frame, (int(x_intersect), int(y_intersect)), 5, (0, 0, 255), -1)
            # # # cv2.circle(frame, (int(x1), h0), 5, (0, 0, 255), -1)
            # # # cv2.circle(frame, (int(x2), h0), 5, (0, 0, 255), -1)

            # # # 定义三角形顶点
            # triangle_pts = np.array([(x_intersect, y_intersect), (x1, self.height), (x2, self.height)], np.int32)

            # cv2.fillPoly(frame, [triangle_pts], (0, 255, 0))
            
################################## 物体检测 ######################################
            out = postprocess(x,
                                anchors, regression, classification,
                                self.regressBoxes, self.clipBoxes,
                                self.threshold, self.iou_threshold)
            out = out[0]
            out['rois'] = scale_coords(frame[:2], out['rois'], shapes[0], shapes[1])
            slope = 0
            target_angle = 0
            for j in range(len(out['rois'])):
                # 获取物体坐标
                x1, y1, x2, y2 = out['rois'][j].astype(int)
                # print("物体坐标", x1, y1, x2, y2)
                box_w = x2 - x1
                "这里添加测距"
                "现实车辆宽度是0.3m, 摄像头中锚框宽度为392像素,距离为0.5米, 锚框宽度224距离是1米"
                "采用相似三角形原理粗略估计距离"
                distance = (self.FocalLength[0]*0.3)/box_w
                # self.ave_w.append(box_w)
                # box_w_ave = sum(self.ave_w) / len(self.ave_w)
                # print("平均锚框宽度为", box_w_ave)
                obj = self.obj_list[out['class_ids'][j]]
                score = float(out['scores'][j])
                if score >= 0.5:
                    plot_one_box(frame, [x1, y1, x2, y2], label=obj, score=score,
                                    color=self.color_list[get_index_label(obj, self.obj_list)], distance=distance)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                    if distance <= 0.8:
                        # 计算屏幕下方中心点
                        screen_center_x, screen_center_y = 640, 719 
                        # # 画屏幕下方中心点
                        # cv2.circle(frame, (int(screen_center_x), int(screen_center_y)), 5, (0, 255, 0), -1)
                        # # 连接物体中心点和屏幕下方中心点并计算斜率
                        cv2.line(frame, (int(center_x), int(center_y)), (640, 719), (255, 0, 0), 2)
                        slope = self.calculate_slope(center_x, center_y, screen_center_x, screen_center_y)
                        # print(slope)           
                        target_angle = self.calculate_angle(abs(slope))
                # 通过斜率判断 向反方向避障
                # 斜率小于> 0 说明物体在右边,那么应该向左避让
                        if target_angle <= 45:
                            if slope < 0:
                                self.N = 1.4
                                angle = angle - 90
                                print("触发左避让,物体在右边")

                            if slope > 0:
                                self.N = 1.4
                                angle = angle + 90
                                print("触发右避让,物体在左边")
            # # 保持最近N个angle值
            # # 在每次计算完angle后进行平滑
            # self.angle_history.append(angle)
            # if len(self.angle_history) > self.N:
            #     self.angle_history.pop(0)  # 保持列表长度为N

            # 计算移动平均
            # smoothed_angle = sum(self.angle_history) / len(self.angle_history)
            angle = max(-50, min(angle, 50))
            angle = int((angle * self.N) + 90)
            angle = self.limit_angle_change(angle, self.previous_angle)
            self.previous_angle = angle
            self.drive(angle, 120)

# # ################################## 自动驾驶 #######################################
#             print(lpos, rpos)
#             frame = self.draw_rectangle(frame, lpos, rpos, offset=Offset)
#             center = (lpos + rpos) / 2
#             angle = int(-(self.width/2 - center))
#             self.drive(angle, 115)
        
            # cv2.imshow("xx", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(1)
            current_time = time.time()
            fps = 1/ (current_time - self.last_time)
            self.last_time = current_time
            self.ave_fps.append(fps)
            print("ave_fps =", sum(self.ave_fps) / len(self.ave_fps))
        # self.udp_socket.sendto(frame_id.to_bytes((frame_id.bit_length() + 7) // 8, byteorder='big'), (self.target_ip, self.target_port)) 
            rtt_msg = UInt32()  
            rtt_msg.data = frame_id
            self.rtt_back.publish(rtt_msg)

def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init(args=args)  # 初始化rclpy
    node = auto_driveSubscriber("ros2_audo_drive_sh")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    # executor = MultiThreadedExecutor()
    # executor.add_node(node)
    # executor.spin()
    node.destroy_node()
    rclpy.shutdown()  # 关闭rclpy


