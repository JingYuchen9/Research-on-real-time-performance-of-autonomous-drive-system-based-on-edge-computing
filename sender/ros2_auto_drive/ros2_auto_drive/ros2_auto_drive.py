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
from sensor_msgs.msg import CompressedImage
from ros2_camera_interface.msg import ImageData, ImageDataArray
import numpy as np
from cv_bridge import CvBridge
from ros2_xycar_interfaces.msg import Motor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类
# from xycar_motor.msg import xycar_motor
# from sensor_msgs.msg import Image
import socket

def signal_handler(sig, frame):
    import time
    time.sleep(2)
    os.system('killall -9 python rosout')
    sys.exit(0)

class auto_driveSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_auto_drive节点被激活")
        self.bridge = CvBridge()
        self.frame_rate = time.time()
        global image
        global pub
        global Width
        global Height
        global Offset
        global Gap

        global frame_count
        frame_count = 0

        image = np.empty(shape=[0])
        pub = None
        Width = 640
        Height = 480
        Offset = 340
        Gap = 40
        qos_profile = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # 创建socket 返回RTT
        self.target_ip = "192.168.31.113"
        self.target_port = 12129
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.camera_subscribe_  = self.create_subscription(ImageData, "ros2_camera", self.img_callback, qos_profile)
        self.motor_publish_ = self.create_publisher(Motor, "XycarMotor", 10)
        # self.timer = self.create_timer(1/30, self.start)

    def img_callback(self, msg):    
        global image
        image = self.bridge.compressed_imgmsg_to_cv2(msg.image_block, "bgr8")
        cv2.imshow('calibration', image)
        cv2.waitKey(1)
        self.udp_socket.sendto(msg.frame_num.to_bytes((msg.frame_num.bit_length() + 7) // 8, byteorder='big'), (self.target_ip, self.target_port))
        self.calcuate_frame_rate()


    # publish xycar_motor msg
    def drive(self, Angle, Speed): 
        global pub
        
        Angle = max(-50, min(Angle, 50))

        Angle = int(Angle * 1.4 + 90)
        # print("Angle", Angle)
        # print("Speed", Speed)
        msg = Motor()
        msg.speed = Speed
        msg.angle = Angle
        self.motor_publish_.publish(msg)

# draw lines
    def draw_lines(self, img, lines):
        global Offset
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.line(img, (x1, y1+ Offset), (x2, y2 + Offset), color, 2)
        return img

    # draw rectangle
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

    # left lines, right lines
    def divide_left_right(self, lines):
        global Width

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

    # get average m, b of lines
    
    def get_line_params(self, lines):
        # sum of x, y, m
        x_sum = 0.0
        y_sum = 0.0
        m_sum = 0.0

        size = len(lines)
        if size == 0:
            return 0, 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)
        m = m_sum / size
        b = y_avg - m * x_avg

        return m, b

    # get lpos, rpos

    def get_line_pos(self, img, lines, left=False, right=False):
        global Width, Height
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

        return img, int(pos)
    
    def get_line_right_pos(self, img, lines, left=False, right=False):
        global Width, Height
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

            cv2.line(img, (int(x1), Height), (int(x2), int((Height/2))), (255, 0,0), 3)

        return img, int(pos)

    # show image and return lpos, rpos
    def process_image(self, frame):
        global Width
        global Offset, Gap

        # gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # blur
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

        # canny edge
        low_threshold = 40
        high_threshold = 50
        edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

        # HoughLinesP
        roi = edge_img[Offset : Offset+Gap, 0 : Width]
        all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,30,10)

        # divide left, right lines
        if all_lines is None:
            return 0, 640
        left_lines, right_lines = self.divide_left_right(all_lines)

        # get center of lines
        frame, lpos = self.get_line_pos(frame, left_lines, left=True)
        frame, rpos = self.get_line_pos(frame, right_lines, right=True)

        # draw lines
        frame = self.draw_lines(frame, left_lines)
        frame = self.draw_lines(frame, right_lines)

        # draw rectangle
        frame = self.draw_rectangle(frame, lpos, rpos, offset=Offset)

        # cv2.imshow('calibration', frame)
        # cv2.waitKey(1)

        return lpos, rpos
    
    def calcuate_frame_rate(self):
        global frame_count
        frame_count += 1
        elapsed_time = time.time() - self.frame_rate
        if elapsed_time >= 1.0:  # 每隔1秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Actual FPS: {fps:.2f}")
            frame_count = 0
            self.frame_rate = time.time()

    def start(self):
        global pub
        global image
        global cap
        global video_mode
        global Width, Height
        
        if image.shape == (480, 640, 3):

            lpos, rpos = self.process_image(image)

            center = (lpos + rpos) / 2
            angle = int(-(Width/2 - center))

            self.drive(angle, 115)

        # rospy.spin()

def main(args=None):

    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init(args=args)  # 初始化rclpy
    node = auto_driveSubscriber("ros2_audo_drive")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown()  # 关闭rclpy


