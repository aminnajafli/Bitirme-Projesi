#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateThroughPoses
from geometry_msgs.msg import PoseStamped

class ParkNavigator(Node):
    def __init__(self):
        super().__init__('park_navigator')
        self._client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        self.timer = self.create_timer(5.0, self.send_goal)
        self._goal_sent = False

    def send_goal(self):
        if self._goal_sent:
            return
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('NavigateThroughPoses hazır değil, bekleniyor...')
            return

        self._goal_sent = True
        self.timer.cancel()

        waypoints = []

        # Waypoint 1
        wp1 = PoseStamped()
        wp1.header.frame_id = 'map'
        wp1.header.stamp = self.get_clock().now().to_msg()
        wp1.pose.position.x = 19.697
        wp1.pose.position.y = -4.279
        wp1.pose.position.z = 0.0
        wp1.pose.orientation.x = 0.0
        wp1.pose.orientation.y = 0.0
        wp1.pose.orientation.z = 0.1109
        wp1.pose.orientation.w = 0.9938
        waypoints.append(wp1)

        # Waypoint 2
        wp2 = PoseStamped()
        wp2.header.frame_id = 'map'
        wp2.header.stamp = self.get_clock().now().to_msg()
        wp2.pose.position.x = 23.205
        wp2.pose.position.y = 9.600
        wp2.pose.position.z = 0.0
        wp2.pose.orientation.x = 0.0
        wp2.pose.orientation.y = 0.0
        wp2.pose.orientation.z = 0.9700
        wp2.pose.orientation.w = 0.2432
        waypoints.append(wp2)

        # Waypoint 3
        wp3 = PoseStamped()
        wp3.header.frame_id = 'map'
        wp3.header.stamp = self.get_clock().now().to_msg()
        wp3.pose.position.x = 8.838
        wp3.pose.position.y = 11.038
        wp3.pose.position.z = 0.0
        wp3.pose.orientation.x = 0.0
        wp3.pose.orientation.y = 0.0
        wp3.pose.orientation.z = 0.7828
        wp3.pose.orientation.w = 0.6223
        waypoints.append(wp3)

        # Waypoint 4 - Park yeri (son hedef)
        wp4 = PoseStamped()
        wp4.header.frame_id = 'map'
        wp4.header.stamp = self.get_clock().now().to_msg()
        wp4.pose.position.x = 6.641
        wp4.pose.position.y = 18.458
        wp4.pose.position.z = 0.0
        wp4.pose.orientation.x = 0.0
        wp4.pose.orientation.y = 0.0
        wp4.pose.orientation.z = 0.7832
        wp4.pose.orientation.w = 0.6217
        waypoints.append(wp4)

        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Waypoint rotası gönderiliyor...')
        self._send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Hedef reddedildi!')
            return
        self.get_logger().info('Rota kabul edildi, araç hareket ediyor...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(
            f'Hedefe kalan mesafe: {dist:.2f} m',
            throttle_duration_sec=2.0
        )

    def result_callback(self, future):
        self.get_logger().info('Park tamamlandi! Arac park yerinde.')

def main(args=None):
    rclpy.init(args=args)
    node = ParkNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
