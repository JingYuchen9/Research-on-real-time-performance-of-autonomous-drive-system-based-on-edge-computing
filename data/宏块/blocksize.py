import matplotlib.pyplot as plt
import numpy as np

# 指数移动平均（EMA）平滑函数
def smooth_data(data, smoothing_factor):
    """
    使用指数移动平均（EMA）平滑数据。

    参数:
    - data: 原始数据，列表或NumPy数组
    - smoothing_factor: 平滑系数，介于0和1之间，越大曲线越平滑

    返回:
    - 平滑后的数据，NumPy数组
    """
    smoothed = np.zeros_like(data)  # 初始化一个与原始数据相同长度的数组
    smoothed[0] = data[0]  # 初始值设为原始数据的第一个点

    # 使用EMA公式逐步计算平滑数据
    for t in range(1, len(data)):
        smoothed[t] = smoothing_factor * data[t] + (1 - smoothing_factor) * smoothed[t - 1]

    return smoothed

# Step 1: 读取文件中的数据
file1 = './blocksize=2/fps.txt'
file2 = './blocksize=4/fps.txt'
file3 = './blocksize=5/fps.txt'
file4 = './blocksize=8/fps.txt'
file5 = './blocksize=10/fps.txt'

# 读取数据
with open(file1, 'r') as f1, open(file2, 'r') as f2, open(file3, 'r') as f3, open(file4, 'r') as f4, open(file5, 'r') as f5:
    data1 = f1.readlines()
    data2 = f2.readlines()
    data3 = f3.readlines()
    data4 = f4.readlines()
    data5 = f5.readlines()

# 数据预处理，将字符串转换为浮点数
data1 = [float(value.strip()) for value in data1]
data2 = [float(value.strip()) for value in data2]
data3 = [float(value.strip()) for value in data3]
data4 = [float(value.strip()) for value in data4]
data5 = [float(value.strip()) for value in data5]

# 找到最短数据的长度，避免插值时出错
min_length = min(len(data1), len(data2), len(data3), len(data4), len(data5))

# 使用最小长度截取每个数据集
data1 = data1[:min_length]
data2 = data2[:min_length]
data3 = data3[:min_length]
data4 = data4[:min_length]
data5 = data5[:min_length]

# 设置平滑参数
smoothing_factor = 1  # 介于0和1之间，越大越平滑

# 对数据进行平滑处理
smoothed_data1 = smooth_data(data1, smoothing_factor)
smoothed_data2 = smooth_data(data2, smoothing_factor)
smoothed_data3 = smooth_data(data3, smoothing_factor)
smoothed_data4 = smooth_data(data4, smoothing_factor)
smoothed_data5 = smooth_data(data5, smoothing_factor)

# 绘制平滑后的折线图
plt.plot(smoothed_data1, color='b', linestyle='-', label='blocksize=2')
plt.plot(smoothed_data2, color='r', linestyle='-', label='blocksize=4')
plt.plot(smoothed_data3, color='g', linestyle='-', label='blocksize=5')
plt.plot(smoothed_data4, color='k', linestyle='-', label='blocksize=8')
# plt.plot(smoothed_data4, color='c', linestyle='-', label='blocksize=10')

# 设置x轴的范围
plt.xlim(0, min_length - 1)

# 添加标题、标签和图例
plt.title('Macro Block Size Definition')
plt.xlabel('Time')
plt.ylabel('Frames Per Second(FPS)')
plt.legend()  # 显示图例

# 显示图表
plt.show()
