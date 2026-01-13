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
        # Parameter for switching gate color
        self.declare_parameter('target_color', 'green')
        
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Point, '/gate/position', 10)
        self.bridge = CvBridge()
        
        # --- CALIBRATION VALUES (REPLACE THESE!) ---
        self.green_lower = np.array([54, 102, 124])
        self.green_upper = np.array([78, 170, 197])
        
        # Red often wraps around 0 and 180 hue
        self.red_lower1 = np.array([3, 44, 77])
        self.red_upper1 = np.array([24, 255, 255])
        self.red_lower2 = np.array([156, 44, 77])
        self.red_upper2 = np.array([180, 255, 255])

    def get_contour(self, hsv, lower, upper):
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500: return largest
        return None

    def image_callback(self, msg):
        target = self.get_parameter('target_color').get_parameter_value().string_value.lower()
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Underwater Preprocessing (CLAHE)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        gate_cnt = None
        if target == 'green':
            gate_cnt = self.get_contour(hsv, self.green_lower, self.green_upper)
        else: # Red
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = mask1 + mask2
            # Find contours on combined mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 500: gate_cnt = largest

        msg_point = Point()
        height, width, _ = frame.shape
        
        if gate_cnt is not None:
            M = cv2.moments(gate_cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Normalize Error (-1 to 1)
                msg_point.x = (cX - (width/2)) / (width/2)
                msg_point.y = (cY - (height/2)) / (height/2)
                msg_point.z = float(cv2.contourArea(gate_cnt))
                
                cv2.drawContours(frame, [gate_cnt], -1, (0, 255, 0), 2)
        
        self.publisher.publish(msg_point)
        cv2.putText(frame, f"TARGET: {target.upper()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Gate View", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(GateVisionNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()