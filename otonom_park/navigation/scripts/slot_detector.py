#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SlotDetector(Node):
    def __init__(self):
        super().__init__('slot_detector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/zed2/left/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, '/slot_detector/debug_image', 10)
        self.get_logger().info('Slot detector başlatıldı')

    def image_callback(self, msg):
        # ROS görüntüsünü OpenCV formatına çevir
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]

        # Sadece alt yarıya bak - park slotları orada
        roi = frame[height//2:, :]

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Beyaz çizgileri tespit et - threshold
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Gürültüyü temizle
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Dikey çizgileri bul (slot ayırıcılar)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, vertical_kernel)

        # Contour bul
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Slot sınırlarını belirle
        slot_boundaries = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 30:  # Yeterince uzun dikey çizgiler
                slot_boundaries.append(x)
                cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

        slot_boundaries.sort()

        # Her slot bölgesini kontrol et
        boş_slotlar = []
        dolu_slotlar = []

        if len(slot_boundaries) >= 2:
            for i in range(len(slot_boundaries) - 1):
                x1 = slot_boundaries[i]
                x2 = slot_boundaries[i+1]
                slot_width = x2 - x1

                if slot_width < 20 or slot_width > 300:
                    continue

                # Slot bölgesini al
                slot_region = gray[:, x1:x2]

                # Slot içinde koyu piksel yoğunluğuna bak
                # Araç varsa daha fazla koyu piksel olur
                dark_pixels = np.sum(slot_region < 80)
                total_pixels = slot_region.size
                dark_ratio = dark_pixels / total_pixels

                slot_center_x = (x1 + x2) // 2
                slot_center_y = roi.shape[0] // 2

                if dark_ratio > 0.6:
                    # Slot boş - zemin görünüyor
                    boş_slotlar.append(i)
                    cv2.putText(roi, f'BOS {i}', (x1+5, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(roi, (x1, 0), (x2, roi.shape[0]-1), (0, 255, 0), 2)
                else:
                    # Slot dolu
                    dolu_slotlar.append(i)
                    cv2.putText(roi, f'DOLU {i}', (x1+5, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(roi, (x1, 0), (x2, roi.shape[0]-1), (0, 0, 255), 2)

        if boş_slotlar:
            self.get_logger().info(f'Boş slotlar: {boş_slotlar}', throttle_duration_sec=2.0)
        if dolu_slotlar:
            self.get_logger().info(f'Dolu slotlar: {dolu_slotlar}', throttle_duration_sec=2.0)

        # Debug görüntüsünü yayınla
        frame[height//2:, :] = roi
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(debug_msg)
	# Direkt pencerede göster
        cv2.imshow('Slot Detector', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SlotDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
