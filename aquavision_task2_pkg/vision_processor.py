#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

class GateVisionNode(Node):
    def __init__(self):
        super().__init__('vision_processor')
        
        # --- THE VARIABLE YOU REQUESTED ---
        # Change this at runtime using: ros2 param set /vision_processor target_color "red"
        self.declare_parameter('target_color', 'green')
        
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Point, '/gate/position', 10)
        self.bridge = CvBridge()
        
        # --- CALIBRATION (UPDATE THESE!) ---
        self.green_lower = np.array([54, 102, 124])
        self.green_upper = np.array([78, 170, 197])
        
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])

        self.kernel = np.ones((5, 5), np.uint8)

    def preprocess_image(self, frame):
        """Standard underwater preprocessing (CLAHE)"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

    def get_largest_contour(self, mask):
        """Finds largest contour in mask after cleaning noise"""
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000: # Threshold for valid object
                return largest
        return None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception: return

        hsv = self.preprocess_image(frame)
        
        # Check the variable setting
        target = self.get_parameter('target_color').get_parameter_value().string_value.lower()
        
        final_mask = None
        if target == 'green':
            final_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        else:
            # Proper Red mixing logic
            m1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            m2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            final_mask = cv2.bitwise_or(m1, m2)

        gate_cnt = self.get_largest_contour(final_mask)
        
        # Output Message
        msg_point = Point()
        height, width, _ = frame.shape
        
        if gate_cnt is not None:
            M = cv2.moments(gate_cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(gate_cnt)
                
                # Normalize Error (-1 to 1)
                msg_point.x = (cX - (width/2)) / (width/2)
                msg_point.y = (cY - (height/2)) / (height/2)
                msg_point.z = float(area) # Z acts as Area/Distance
                
                # Draw Box
                x,y,w,h = cv2.boundingRect(gate_cnt)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, (cX, cY), 5, (0,0,255), -1)

        self.publisher.publish(msg_point)
        
        cv2.putText(frame, f"TARGET: {target.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Gate Vision", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GateVisionNode())
    rclpy.shutdown()