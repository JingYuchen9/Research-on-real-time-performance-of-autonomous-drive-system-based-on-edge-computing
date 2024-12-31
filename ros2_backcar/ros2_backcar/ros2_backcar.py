# 当侧面雷达检测到物体时开始倒车,当后面雷达检测到物体时停车
import rclpy
from std_msgs.msg import UInt32MultiArray
from ros2_xycar_interfaces.msg import Motor
from rclpy.node import Node
import time

class backcarSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("backcarSubscriber节点被激活")
        self.start_time = 0
        self.ultrasonic_subscribe_  = self.create_subscription(UInt32MultiArray, "ros2_ultrasonic", self.backcarHandle, 10)
        self.motor_publish_ = self.create_publisher(Motor, "XycarMotor", 10)
        self.timer = self.create_timer(0.5, self.drive_motor)

# array('I', [76, 200, 184, 200, 93, 74, 72, 49])
# array('I', [75, 200, 184, 200, 94, 74, 73, 51])
# array('I', [75, 200, 184, 200, 93, 76, 72, 49])
# array('I', [69, 200, 184, 200, 93, 74, 72, 53])
 
    def backcarHandle(self, msg):
        # 检测侧面的物体距离
        # 直行
        if msg.data[4] > 20 and msg.data[6] > 30:
            self.Angle = 90
            self.Speed = 115
        
        # 停靠
        if msg.data[4] <= 20 and msg.data[6] > 30:
            if self.start_time == 0:
                self.start_time = time.time()
            if time.time() - self.start_time <= 2:
                self.Angle = 90
                self.Speed = 90
            if time.time() - self.start_time > 2:
                print("检测到标志,开始倒车...")
                self.Angle = 160
                self.Speed = 70
                if msg.data[5] == msg.data[7]:
                    self.Angle = 90
                    self.Speed = 70
        if msg.data[6] <= 20:
            self.Angle = 90
            self.Speed = 90

    def drive_motor(self):
        msg = Motor()
        msg.speed = self.Speed
        msg.angle = self.Angle
        self.motor_publish_.publish(msg)
        print("angle = ", msg.angle, "speed = ", msg.speed)

def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = backcarSubscriber("ros2_backcar")  # 新建一个节点
    rclpy.spin(node)  # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    node.cap.release()
    rclpy.shutdown()  # 关闭rclpy