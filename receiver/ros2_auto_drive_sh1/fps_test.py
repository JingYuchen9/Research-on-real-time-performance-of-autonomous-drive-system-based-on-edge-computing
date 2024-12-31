import os
# import io

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import CCNet as Network
import torch
import torchvision.transforms as transforms
from PIL import Image

# def color_adjust(img):
#     B, G, R, A = cv2.split(img)
#     img=cv2.merge((B, G, R))
#     return img

def color_adjust(img):
    B,G,R = cv2.split(img)
    img=cv2.merge([B,G,R])
    return img


def get_array_inverse(arr1):
    return np.where(arr1 == 0, 1, 0)

def seg_mask_im_macro(arr2, block_size=8):
    arr1 = np.array(arr2)
    h, w = arr1.shape
    block_id = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_id += 1
            block = arr1[i:i+block_size, j:j+block_size]
            if np.any(block==0):
                arr1[i:i+block_size, j:j+block_size] = 0
    return arr1

def overlay_grayscale_on_color(color_frame, grayscale_frame):
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

def merge_image(frame):

    # 应用预处理
    input_tensor = preprocess(frame).to(device)

    # 增加 batch 维度
    input_batch = input_tensor.unsqueeze(0)  # 形状变为 (1, 3, H, W)

    with torch.no_grad():
        segresult = model(input_batch)

    segresulti = segresult[0,0,:,:,]*0+segresult[0,1,:,:,]*1+segresult[0,2,:,:,]*2+segresult[0,3,:,:,]*3
    # io.imsave('./seg'+"haha"+'.png', (segresulti.data.cpu().numpy() * 255).astype(np.uint8))
    # 转换为numpy数组

    segresulti_np = segresulti.data.cpu().numpy()

    # 将segresulti缩放到0-255并转换为uint8类型
    segresulti_np = (segresulti_np * 255).astype(np.uint8)
    # arr_im = get_array_inverse(segresulti_np)
    # seg_mask_im_macro_old = seg_mask_im_macro(arr_im, block_size=20)
    # im = np.array(seg_mask_im_macro_old)
    # seg_mask_uim_macro = np.expand_dims(im,2).repeat(3,axis=2)
    # im_images = seg_mask_uim_macro * frame
    # img = color_adjust(im_images)

    # uim = get_array_inverse(im)
    # seg_mask_uim_macro = np.expand_dims(uim,2).repeat(3,axis=2)
    # uim_images = seg_mask_uim_macro * frame
    # return img.astype(np.uint8)


    image = overlay_grayscale_on_color(frame, segresulti_np)
    # image = color_adjust(image)
    # b, g, r, a = cv2.split(image)
    # image = cv2.merge((b, g, r))
    # high, width, _ = segresulti_np.shape
    # block_size = 80
    # block_id = 0
    # for y in range(0, high, block_size):
    #         for x in range(0, width, block_size):
    #             block_id += 1
    #             block = image[y:y + block_size, x:x + block_size]
    #             cv2.putText(block, str(block_id), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    #             # 计算符合条件的像素数量
    #             white_pixels = np.sum(np.all(block == (255, 255, 255), axis=-1))

    #             # # 计算白色像素占总像素的百分比
    #             total_pixels = block.shape[0] * block.shape[1]
    #             if white_pixels / total_pixels > 0.05:
    #                 ori.append(block_id)
    #             # 用ros2 发走
    return image

def main():
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("无法接收视频流，退出。")
            break

        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 0]  # 压缩率10 占用带宽500kb以内，压缩率90 占用带宽最高2-3M
        # _, encoding = cv2.imencode('.jpg', frame, encode_param)  # 开始压缩 上面使用了JPGE压缩算法
        # frame = cv2.imdecode(np.frombuffer(encoding, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)), interpolation=cv2.INTER_AREA)
        image = merge_image(frame)

        # 使用OpenCV显示图像
        cv2.imshow("Segmented Result", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("spend time", time.time() - start_time)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # cap = cv2.VideoCapture(cv2.CAP_V4L2)
    cap = cv2.VideoCapture('/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/1.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    ori = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    model = Network.LETNet().to(device)
    model.load_state_dict(torch.load("/home/jing/ros2_ws/src/ros2_auto_drive_sh1/ros2_auto_drive_sh1/pkl/letnet.pkl", weights_only=True))
    # model.load_state_dict(torch.load('./pkl/baseline.pth'))
    model.eval()
    main()