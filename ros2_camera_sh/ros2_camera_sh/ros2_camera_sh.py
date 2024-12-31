#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
import socket
import struct
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt32
# from ros2_camera_interface.msg import ImageData
import concurrent.futures
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # ROS2 QoS类
from ros2_camera_interface.msg import ImageData
import cv2
import numpy as np
import threading
import queue
import time
    
class cameraPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("ros2_camera_sh节点被激活")

        # 获取socket信息
        self.local_ip  = "192.168.123.88"
        self.local_port = 12124
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.local_ip, self.local_port))

        # 视频解析
        self.total_data_dict = dict()
        self.total_data_list = []

        # 计算延迟
        self.RTT_dict = dict()
        self.rtt_list = []
        self.image_tuple = tuple()

        # 帧编号
        self.sum = 0

        # 数据分片大小
        self.block_size = 20

        # 设置分割字符
        self.target_bytes = b"\xff\xd9"
        
        self.point_frame = 1

        # 图像传输
        self.bridge = CvBridge()

        # 创建缓存
        self.buffer_dict = dict()
        self.buffer_queue = queue.Queue()

        # 创建新线程
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.player, args=(self.event, ))
        self.thread.start()  # 启动处理线程

        self.event_send = threading.Event()
        self.thread_send = threading.Thread(target=self.publish_msg, args=(self.event_send, ))
        self.thread_send.start()  # 启动发布线程

        # self.thread_get = threading.Thread(target=self.get_camera_param)
        # self.thread_get.start()  # 启动发布线程

        # 创建消息
        self.command_publisher_ = self.create_publisher(ImageData, "ros2_camera_sh", 10)
        # self.RTT_subscribe_ = self.create_subscription(UInt32, "rtt", self.rtt_back, 10)

        # 计算帧率
        global frame_count
        frame_count = 0
        self.frame_rate = time.time()
        # 运行主程序
        self.get_camera_param()
        
    def get_camera_param(self):
        # point = 1
        # 从底层获取图像数据
        while True:
            rtp_packet, address = self.udp_socket.recvfrom(2048)
            # 解析RTP头部
            (rqt_cc, rqt_payload_type, rqt_sequence_number, rqt_timestamp, rqt_ssrc) = struct.unpack('!BBHLL', rtp_packet[:12])
            # if rqt_sequence_number > point:
            #     self.RTT_dict[point] = time.time()
            #     point = rqt_sequence_number
            payload = self.get_payload(rtp_packet)
            # self.buffer_queue.put(rtp_packet)
            self.buffer_dict[rqt_ssrc] = rtp_packet
            if self.find_bytes_from_payload(payload) >= 0:
                packet_dict_sorted = self.sort_dict_by_keys(self.buffer_dict)
                for value in packet_dict_sorted.values():
                    self.buffer_queue.put(value)
                self.event.set()
                self.buffer_dict.clear()

    def player(self,event):
        print("子线程开始运行")
        self.event.wait()
        while True:
            rtp_packet = self.buffer_queue.get()
            (rqt_cc, rqt_payload_type, rqt_sequence_number, rqt_timestamp, rqt_ssrc) = struct.unpack('!BBHLL', rtp_packet[:12])

            payload = self.get_payload(rtp_packet)
            if rqt_sequence_number == self.point_frame:
                self.total_data_dict[rqt_ssrc] = payload

            if rqt_sequence_number > self.point_frame:
                sorted_keys = sorted(self.total_data_dict.keys())  # 对字典中的key进行排序，返回一个有序的list 里面都是key
                sorted_values = [self.total_data_dict[key] for key in sorted_keys]  # 遍历字典根据key取出对应的value
                self.total_data_dict.clear()
                combined_bytes = b''.join(sorted_values)
                frame_data = cv2.imdecode(np.frombuffer(combined_bytes, np.uint8), cv2.IMREAD_COLOR)
                if isinstance(frame_data, np.ndarray):
                    publish_image = self.bridge.cv2_to_compressed_imgmsg(frame_data, dst_format = "jpeg")
                    self.image_tuple = (self.point_frame, publish_image)
                    self.event_send.set()
                    # print("播放中...")
                    # self.get_logger().info('Published image message.')
                    # if frame_data is not None:
                    #     cv2.imshow('Image', frame_data)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                    if frame_data is None:
                        print("解码失败, 发生丢包!")
                if not isinstance(frame_data, np.ndarray):
                    print("frame_data 不是 np.ndarray类型!!!!!")
                    continue
                self.point_frame = rqt_sequence_number
                self.total_data_dict[rqt_ssrc] = payload

            if rqt_sequence_number < self.point_frame:
                print("错乱帧")
                continue

    def publish_msg(self, event):
        self.event_send.wait()
        while True:
            msg = ImageData()
            msg.frame_num = self.image_tuple[0]
            msg.block_num = 0
            msg.total_block_num = 0
            msg.image_block = self.image_tuple[1]
            self.command_publisher_.publish(msg)
            time.sleep(1/30)
            self.calcuate_frame_rate() # 计算帧率

    def rtt_back(self, msg):
        num = msg.data
        current_time = time.time()
        send_time = self.RTT_dict[num]
        RTT_ = (current_time - send_time)
        self.rtt_list.append(RTT_)
        RTT_num = len(self.rtt_list)
        if RTT_num >= 30:
            ave = sum(self.rtt_list) / RTT_num
            print("一秒内平均RTT", ave/2)
            self.rtt_list.clear()


    # 获取有效载荷
    @staticmethod
    def get_payload(packet):
        payload = packet[12:]
        return payload
    
    # 排序字典返回新字典
    @staticmethod
    def sort_dict_by_keys(input_dict):
        # 使用 sorted 函数按照字典键进行排序
        sorted_items = sorted(input_dict.items())

        # 使用 collections 模块的 OrderedDict 类构建有序字典
        sorted_dict = dict(sorted_items)

        return sorted_dict
    
    # 计算帧率
    def calcuate_frame_rate(self):
        global frame_count
        frame_count += 1
        elapsed_time = time.time() - self.frame_rate
        if elapsed_time >= 1.0:  # 每隔1秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Actual FPS: {fps:.2f}")
            frame_count = 0
            self.frame_rate = time.time()

    # 读取收到payload，搜索是否包含b"\xff\xd9" 如果包含，则将前面所有的值存入，前一帧后面的值存入后一帧
    def find_bytes_from_payload(self, payload):
        index = payload.find(self.target_bytes)
        return index
        
def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = cameraPublisher("ros2_camera_sh")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    node.udp_socket.close()
    print("shutdown!!!")
    rclpy.shutdown()  # 关闭rclpy