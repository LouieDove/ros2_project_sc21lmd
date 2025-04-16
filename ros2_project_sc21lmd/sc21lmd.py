import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import sin, cos, sqrt, atan2

class NavigationGoalActionClient(Node):
    def __init__(self):
        super().__init__('navigation_goal_action_client')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.bridge = CvBridge()
        self.blue_box_coord = None  # Store blue box coordinates
        self.current_pose = None  # Store current robot pose
        self.is_final_goal = False
        self.shutting_down = False  # Prevent multiple shutdowns

        # Path
        self.path = [
            (-0.3, -5.5, 0.0),
            (3.0, -6.0, 0.0),
            (3.0, -11.0, 0.0),
            (-3.0, -12.0, 0.0),
            (-8.0, -7.0, 0.0),
            (-8.0, 0.0, 0.0)
        ]
        self.path_index = 0

        # Known box coordinates
        self.box_coords = [
            (5.0, -7.5, 0.0),
            (-3.5, -9.0, 0.0),
            (-7.0, -0.2, 0.0)
        ]

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        if not self.action_client.wait_for_server(timeout_sec=10.0):
            return

        self.send_goal_future = self.action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        if self.is_final_goal:
            self.get_logger().info("Reached final goal near blue box")
            self.create_timer(5.0, self.shutdown)
            return
        elif self.path_index < len(self.path) - 1:
            self.path_index += 1
            x, y, yaw = self.path[self.path_index]
            self.send_goal(x, y, yaw)
        elif self.blue_box_coord is not None:
            box_x, box_y = self.blue_box_coord
            current_x, current_y = self.path[-1][:2]
            dx = box_x - current_x
            dy = box_y - current_y
            norm = sqrt(dx**2 + dy**2)
            if norm > 1.0:
                new_x = current_x + (dx / norm) * (norm - 1.0)
                new_y = current_y + (dy / norm) * (norm - 1.0)
            else:
                new_x, new_y = current_x, current_y
            yaw = atan2(dy, dx)  # Face the blue box
            self.is_final_goal = True
            self.send_goal(new_x, new_y, yaw)
        else:
            self.get_logger().info("No blue box detected, shutting down")
            self.create_timer(5.0, self.shutdown)

    def shutdown(self):
        if not self.shutting_down:
            self.shutting_down = True
            self.get_logger().info("Shutting down")
            self.destroy_node()
            rclpy.shutdown()

    def image_callback(self, data):
        if self.current_pose is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception:
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        blue_lower = np.array([100, 150, 0])
        blue_upper = np.array([140, 255, 255])

        red_mask = cv2.inRange(hsv_image, red_lower1, red_upper1) | cv2.inRange(hsv_image, red_lower2, red_upper2)
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        for mask, color, name in [(red_mask, (0, 0, 255), "Red"),
                                  (green_mask, (0, 255, 0), "Green"),
                                  (blue_mask, (255, 0, 0), "Blue")]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                self.get_logger().info(f"{name} detected with {len(contours)} contours")
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        if name == "Blue":
                            min_dist = float('inf')
                            closest_box = None
                            for box in self.box_coords:
                                dist = sqrt((self.current_pose.position.x - box[0])**2 + (self.current_pose.position.y - box[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_box = box
                            if min_dist < 5.0:
                                new_coord = closest_box[:2]
                                if self.blue_box_coord != new_coord:
                                    self.blue_box_coord = new_coord
                                    self.get_logger().info(f"Blue box coordinates updated: {self.blue_box_coord}")

        cv2.imshow('Color Detection', image)
        cv2.waitKey(1)

    def start_navigation(self):
        if self.path:
            x, y, yaw = self.path[self.path_index]
            self.send_goal(x, y, yaw)

def main(args=None):
    rclpy.init(args=args)
    navigation_goal_action_client = NavigationGoalActionClient()
    navigation_goal_action_client.start_navigation()
    try:
        rclpy.spin(navigation_goal_action_client)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    navigation_goal_action_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()