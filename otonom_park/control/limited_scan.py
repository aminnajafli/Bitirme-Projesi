#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from builtin_interfaces.msg import Time

class LimitedScan(Node):
    def __init__(self):
        super().__init__('limited_scan')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # RPLIDAR'ın yayın yaptığı topic genelde /scan
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(
            LaserScan,
            '/limited_scan',  # lidar_dur.py ile eşleşen topic adı
            10)

        self.declare_parameter('setting_angle', 30.0)
        self.setting_angle = self.get_parameter('setting_angle').get_parameter_value().double_value

    def listener_callback(self, msg):
        # Robotun önü için 60° açı (±30°): sadece önde engel tespiti
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        total_angles = len(msg.ranges)

        start_angle = np.radians(-self.setting_angle)
        end_angle = np.radians(self.setting_angle)

        start_idx = int((start_angle - angle_min) / angle_increment)
        end_idx = int((end_angle - angle_min) / angle_increment)
        
        # Index sınırlarını kontrol et
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, total_angles)
        

        # Sadece bu aralığı seç
        new_ranges = msg.ranges[start_idx:end_idx] if end_idx > start_idx else []
                # Yeni bir LaserScan mesajı oluştur
        limited_scan = LaserScan()
        limited_scan.header = msg.header
        # Update timestamp to current time to avoid transform timing issues
        limited_scan.header.stamp = self.get_clock().now().to_msg()
        # Keep the original frame_id from the source laser scan
        # limited_scan.header.frame_id = "laser"  # Don't override the original frame_id
        limited_scan.angle_min = start_angle
        limited_scan.angle_max = end_angle
        limited_scan.angle_increment = angle_increment
        limited_scan.time_increment = msg.time_increment
        limited_scan.scan_time = msg.scan_time
        limited_scan.range_min = msg.range_min
        limited_scan.range_max = msg.range_max
        limited_scan.ranges = new_ranges
        
        # Intensities varsa aynı aralığı al
        if len(msg.intensities) > 0:
            limited_scan.intensities = list(msg.intensities[start_idx:end_idx])
        else:
            limited_scan.intensities = []

        self.publisher.publish(limited_scan)

def main(args=None):
    rclpy.init(args=args)
    node = LimitedScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
