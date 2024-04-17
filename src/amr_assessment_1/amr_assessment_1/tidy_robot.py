import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from collections import deque

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class TidyUpRobot(Node):     
    def __init__(self):
        super().__init__('tidy_up_robot')
        self.image_subscriber = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.twist = Twist()
        self.cube_positions = deque()
        self.image_width = None
        self.image_height = None
        self.last_cube_position = None
        self.is_moving = False

    def publish_velocity(self, linear_velocity, angular_velocity):
        self.twist.linear.x = float(linear_velocity)
        self.twist.angular.z = float(angular_velocity)
        self.cmd_vel_publisher.publish(self.twist)


    def detect_objects(self, image, image_width, image_height):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the color range for the brightly colored objects
        lower_color = np.array([40, 40, 40]) # Example: lower range for bright colors
        upper_color = np.array([70, 255, 255]) # Example: upper range for bright colors

        # Create a mask for the color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Add the cube's position to the queue
                self.cube_positions.append((cX, cY))

    def move_towards_next_object(self):
        if self.cube_positions:
            # Pop the next cube's position from the queue
            x, y = self.cube_positions.popleft()
            self.move_towards_object(x, y, self.image_width, self.image_height)
            self.is_moving = True
        else:
            # If there are no more cubes, stop the robot
            self.publish_velocity(0, 0)
            self.is_moving = False
    
    def move_towards_object(self, x, y, image_width, image_height):
        # Calculate the direction vector from the robot to the cube
        direction_vector = np.array([x - image_width / 2, y - image_height / 2])

        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the angle to the object relative to the robot's current orientation
        # Assuming the robot's current orientation is aligned with the x-axis
        angle_to_object = np.arctan2(direction_vector[1], direction_vector[0])

        # Calculate the distance to the object
        distance_to_object = np.sqrt((x - image_width / 2)**2 + (y - image_height / 2)**2)

        # Initialize the PID controller
        pid_controller = PIDController(kp=0.5, ki=0.1, kd=0.05)

        # Update the PID controller
        angular_velocity = pid_controller.update(angle_to_object, 0.1) # Assuming a fixed time step of 0.1 seconds

        # Adjust the robot's speed based on the distance to the object
        linear_velocity = 0.5 if distance_to_object > 100 else 0.2 # Example: slow down when close

        # Publish the velocity command
        self.publish_velocity(linear_velocity, angular_velocity)

    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Store the image width and height as attributes
        self.image_width = msg.width
        self.image_height = msg.height
    
        # Process the OpenCV image as needed
        # For example, detect objects in the image
        self.detect_objects(cv_image, msg.width, msg.height)

        # Continuously move towards the next object until no more cubes are detected
        while self.cube_positions:
            self.move_towards_next_object()
            rclpy.spin_once(self, timeout_sec=0.1) # Allow other callbacks to run
    
    def rotate(self, angular_velocity, duration):
        """Rotate the robot for a specified duration."""
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        while self.get_clock().now().seconds_nanoseconds()[0] - start_time < duration:
            self.publish_velocity(0, angular_velocity)
            rclpy.spin_once(self, timeout_sec=0.1)

    def scan_callback(self, msg):
        min_distance = 0.5 # Minimum safe distance from obstacles
        for range in msg.ranges:
            if range < min_distance:
                # If the robot is against a wall, rotate to scan for cubes behind it
                self.rotate(angular_velocity=1.5, duration=2) # Rotate at 1.5 rad/s for 2 seconds
                return
            
    def main_loop(self):
        while rclpy.ok():
            if not self.is_moving:
                # If the robot is not currently moving towards a cube, check for new cubes
                self.move_towards_next_object()
            rclpy.spin_once(self, timeout_sec=0.1)
        

def main(args=None):
    rclpy.init(args=args)
    tidy_up_robot = TidyUpRobot()
    tidy_up_robot.main_loop() # Start the main loop
    tidy_up_robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
