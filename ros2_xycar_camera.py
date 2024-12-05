#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import struct
import socket
import threading
import rclpy
from rclpy.node import Node
import concurrent.futures
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from ros2_camera_interface.msg import ImageData, ImageDataArray, ImageNoCompress
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
import numpy as np
from std_msgs.msg import UInt32
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类

class ros2_xycar_camera(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_xycar_camera节点被激活")
        # 计算RTT
        # self.local_ip  = "192.168.31.152"
        # # self.local_ip  = "10.30.115.50"
        # self.local_port = 12129
        # self.RTT_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.RTT_udp_socket.bind((self.local_ip, self.local_port))
        self.rtt_dict = dict()
        self.rtt_list = []
        self.rtt = 0
        cb_group1 = ReentrantCallbackGroup()# ReentrantCallbackGroup()
        cb_group2 = ReentrantCallbackGroup()
        cb_group3 = MutuallyExclusiveCallbackGroup()

        # 创建新线程接收RTT
        # self.RTT_thread = threading.Thread(target=self.RTT_rec)
        # self.RTT_thread.daemon = True
        # self.RTT_thread.start()  # 启动守护线程

        self.bridge = CvBridge()
        self.fps = 38
        qos_profile1 = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile2 = QoSProfile(     # 创建一个QoS原则
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        self.image_publish_ = self.create_publisher(ImageData, "ros2_xycar_camera",qos_profile1, callback_group=cb_group1)

        self.rtt_back = self.create_subscription(UInt32, "rtt", self.auto_drive_callback, qos_profile2, callback_group=cb_group2)
        # self.Compressed_thread = threading.Thread(target=self.compress)
        # self.Compressed_thread.start()
        self.get_frame = self.create_timer(1/30, self.start, callback_group=cb_group3)
        global frame_count
        frame_count = 0
        self.frame_rate = time.time()
   
        # 摄像头参数设置
        self.cap = cv2.VideoCapture(cv2.CAP_V4L2)
        # self.cap = cv2.VideoCapture('/home/jing/ros2_ws/src/ros2_xycar_camera/ros2_xycar_camera/.1.mp4')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 帧率计算
        self.time_rate = 0
        self.sum = 0
        self.starttime = time.time()

        # 打印延时
        self.rtt_file = open('RTT.txt', 'w')
        self.ave_rtt_file = open('ave_RTT.txt', 'w')

        # 保存视频
        # self.video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

    def start(self):
        ret, frame = self.cap.read()
        if not ret:
            print("无法接收视频流，退出。")
        self.sum += 1
        self.rtt_dict[self.sum] = time.time()
        # self.video_writer.write(frame)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 6M左右
        # _, encoding = cv2.imencode('.jpg', frame, encode_param) 
        # frame = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format = "jpeg")
        # compressed_msg = CompressedImage()
        # compressed_msg.data = np.array(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).tobytes()
        msg = ImageData()
        msg.frame_num = self.sum
        msg.block_num = 0
        msg.total_block_num = 0
        msg.image_block = compressed_msg

        # image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        # msg = ImageNoCompress()
        # msg.frame_num = self.sum
        # msg.image = image
        # 设置10分钟后停止用于测试
        if time.time() - self.starttime <= 900:
            self.image_publish_.publish(msg)


    def auto_drive_callback(self, msg):
        frame_id = msg.data
        del msg
        send_time = self.rtt_dict.pop(frame_id, False)
        if not send_time:
            print("不存在")
            return
        self.rtt = (time.time() - send_time)/2
        self.rtt_list.append(self.rtt)
        print("RTT:", self.rtt)
        self.rtt_file.write(f"{self.rtt:.6f}\n")
        # 计算并打印平均RTT
        avg_rtt = sum(self.rtt_list) / len(self.rtt_list)
        print("Average RTT:", avg_rtt)
        self.ave_rtt_file.write(f"{avg_rtt:.6f}\n")
        


        # while True: 
        #     # time.sleep(1/self.fps)
        #     ret, frame = self.cap.read()
        #     if not ret:
        #         print("无法接收视频流，退出。")
        #         break
        #     self.sum += 1
        #     self.rtt_dict[self.sum] = time.time()
        #     # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 6M左右
        #     # _, encoding = cv2.imencode('.jpg', frame, encode_param) 
        #     # frame = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        #     compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format = "jpeg")
        #     # compressed_msg = CompressedImage()
        #     # compressed_msg.data = np.array(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).tobytes()
        #     msg = ImageData()
        #     msg.frame_num = self.sum
        #     msg.block_num = 0
        #     msg.total_block_num = 0
        #     msg.image_block = compressed_msg
        #     # image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        #     # msg = ImageNoCompress()
        #     # msg.frame_num = self.sum
        #     # msg.image = image
        #     self.image_publish_.publish(msg)
        #     # 设置10分钟后停止用于测试
        #     if time.time() - self.starttime >= 300:
        #         break
        #     # print(1 / (time.time() - self.starttime))
        #     # self.starttime = time.time()
        #     # self.calcuate_frame_rate() 50帧差不多

    # def RTT_rec(self):
    #     try:
    #         with open("RTT.txt", 'w') as rtt_f, open("ave_RTT.txt", 'w') as avg_rtt_f:
    #             while True:
    #                 RTT_bytes, address = self.RTT_udp_socket.recvfrom(1024)
    #                 # 确定字节串长度
    #                 byte_length = len(RTT_bytes)
    #                 frame_id = struct.unpack('>{}B'.format(byte_length), RTT_bytes)

    #                 # 将元组转换为整数
    #                 frame_id = int(''.join(map(lambda x: format(x, '02x'), frame_id)), 16)
    #                 # print("frame_id = ", frame_id)
    #                 send_time = self.rtt_dict.pop(frame_id, False)
    #                 if not send_time:
    #                     continue
    #                 rtt = time.time() - send_time
    #                 self.rtt_list.append(rtt)
    #                 print("RTT:", rtt)
    #                 rtt_f.write(f"{rtt:.6f}\n")

    #                 # 计算并打印平均RTT
    #                 avg_rtt = sum(self.rtt_list) / len(self.rtt_list)
    #                 print("Average RTT:", avg_rtt)
    #                 avg_rtt_f.write(f"{avg_rtt:.6f}\n")

    #     except KeyboardInterrupt:
    #         self.RTT_udp_socket.close()

    def calcuate_frame_rate(self):
        global frame_count
        frame_count += 1
        elapsed_time = time.time() - self.frame_rate
        if elapsed_time >= 1.0:  # 每隔1秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Actual FPS: {fps:.2f}")
            frame_count = 0
            self.frame_rate = time.time()


def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = ros2_xycar_camera("ros2_xycar_camera")  # 新建一个节点
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    # rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown()  # 关闭rclpy
    if node.video_writer.isOpened():
            node.video_writer.release()


if __name__ == "__main__":
    main()
