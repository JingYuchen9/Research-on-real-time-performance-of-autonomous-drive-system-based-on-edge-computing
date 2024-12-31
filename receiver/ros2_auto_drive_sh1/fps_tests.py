#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from collections import deque
import CCNet as Network  # 假设 CCNet 定义了 MiniNetv2 模型

class MiniNetv2FPS_Test:
    def __init__(self, video_path, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载模型
        self.model = Network.LETNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()

        # 视频读取
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # 记录帧率
        self.fps_log = []
        self.start_time = time.time()

    def process_frame(self, frame):
        # 预处理输入帧
        input_tensor = self.preprocess(frame).to(self.device)
        input_batch = input_tensor.unsqueeze(0)  # 增加 batch 维度

        # 模型推理
        with torch.no_grad():
            output = self.model(input_batch)

        # 后处理
        seg_result = output[0].argmax(0).cpu().numpy()  # 获取每像素分类结果
        return seg_result

    def run(self, duration=300):
        frame_count = 0
        total_frames = 0
        start_time = time.time()
        second_start_time = start_time

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                print("视频读取结束")
                break

            # 推理开始时间
            start_infer = time.time()
            _ = self.process_frame(frame)
            end_infer = time.time()
            frame_count += 1
            total_frames += 1

            # 如果一秒过去了，计算帧率
            if end_infer - second_start_time >= 1.0:
                print(f"当前秒内推理帧数: {frame_count}")
                second_start_time = end_infer
                frame_count = 0

        # 输出总平均帧率
        avg_fps = total_frames / (time.time() - start_time)
        print(f"测试完成，总帧数: {total_frames}, 平均帧率: {avg_fps:.2f}")

        # 释放资源
        self.cap.release()

if __name__ == "__main__":
    # 设置视频路径和模型路径
    video_path = "/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/1.mp4"
    model_path = "/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/pkl/letnet.pkl"

    # 初始化测试类
    fps_tester = MiniNetv2FPS_Test(video_path, model_path)

    # 运行帧率测试
    fps_tester.run(duration=300)
