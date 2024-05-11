import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Float32, Int32MultiArray
from nav_msgs.msg import Odometry
import math
import numpy as np

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.initialized_traj = False

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")
        self.pose_array_traj = PoseArray
        self.traj_xy = np.zeros((1,2))
        self.trajxy_robot_frame = np.zeros((1,2))

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.scan_sub = self.create_subscription(PoseArray,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.laser_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.store_laser_scan,
            10)
        
        self.path_pub = self.create_publisher(Marker, "/path_plan", 1)

        self.checkpoints_sub = self.create_subscription(PoseArray, "/checkpoints", self.checkpoints_callback, 1)

        self.subscription = self.create_subscription(
            Int32MultiArray,
            "/checkpoints/indexes",
            self.checkpoint_indices_callback,
            10
        )

        self.stop_sign_sub = self.create_subscription(Float32, "stop_area", self.stop_sign_callback, 10)

        self.laser_scan = LaserScan()
        self.laserx = np.zeros(1)
        self.lasery = np.zeros(1)

        self.relative_x = 1.
        self.relative_y = 0.

        self.current_robot_x = 0
        self.current_robot_y = 0
        self.current_robot_theta = 0
        self.driving_forward = True

        self.switch_to_drive_back_thresh = 0.5
        self.switch_to_drive_frwrd_thresh = 0.75

        self.checkpoint_indexes = []
        self.checkpoint_poses = []

        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer_seconds_remaining = 3
        self.timer_look_for_stop_signs = 0
        self.stop_sign_size_threshold = 100

        
    def pose_callback(self, odometry_msg):
        #self.get_logger().info("running pose callback")
        #raise NotImplementedError
        self.latest_odom = odometry_msg
        self.current_velocity = self.latest_odom.twist.twist.linear.x

        self.current_robot_x = self.latest_odom.pose.pose.position.x
        self.current_robot_y = self.latest_odom.pose.pose.position.y

        orient = self.latest_odom.pose.pose.orientation
        self.current_robot_theta = self.quaternion_to_yaw(orient.x, orient.y, orient.z, orient.w)

        self.trajxy_robot_frame = self.global_to_local(self.traj_xy, self.current_robot_x, self.current_robot_y, self.current_robot_theta)

        # self.get_logger().info(f"running pose callback {str(self.relative_x), self.relative_y}")

        self.recompute_optimal_controls()

    def stop_sign_callback(self, msg):
        if self.timer_look_for_stop_signs == 0:
            stopSignArea = msg.data
            if stopSignArea > self.stop_sign_size_threshold:
                self.timer_seconds_remaining = 3
                self.timer_look_for_stop_signs = 8


    def timer_callback(self):
        self.timer_seconds_remaining = max(0, self.timer_seconds_remaining - 1)
        self.timer_look_for_stop_signs = max(0, self.timer_look_for_stop_signs - 1)
        #self.get_logger().info(f"time on timer: {self.timer_seconds_remaining}")

    def checkpoint_indices_callback(self, msg):
        self.checkpoint_indexes = list(msg.data)
        #self.get_logger().info(f"checkpoint indexes: {self.checkpoint_indexes}")
    
    def recompute_optimal_controls(self):
        if len(self.checkpoint_indexes) == 0:
            return

        index_closest_point = 0
        closest_dist = 99999
        found_point_in_lookahead = False

        # self.checkpoint_indexes = []
        # for pose in self.checkpoint_poses:
        #     distances = np.abs(self.traj_xy[:,0] - pose.position.x) + np.abs(self.traj_xy[:,1] - pose.position.y)
        #     self.checkpoint_indexes.append(np.argmin(distances))

        #self.get_logger().info(f"checkpoint ind: {self.checkpoint_indexes}")

        for i in range(self.checkpoint_indexes[0], 0, -1):
            #for finding the closest point
            if abs(self.trajxy_robot_frame[i, 0]) + abs(self.trajxy_robot_frame[i, 1]) < closest_dist:
                index_closest_point = i
                closest_dist = abs(self.trajxy_robot_frame[i, 0]) + abs(self.trajxy_robot_frame[i, 1])

            if abs(self.trajxy_robot_frame[i, 0]) < self.lookahead and abs(self.trajxy_robot_frame[i, 1]) < self.lookahead:
                self.relative_x, self.relative_y = self.trajxy_robot_frame[i]
                index_closest_point = i
                found_point_in_lookahead = True
                break
            #if it didn't find a point close enough
        
        if not found_point_in_lookahead and self.trajxy_robot_frame.shape[0] > 0:
            self.relative_x, self.relative_y = self.trajxy_robot_frame[index_closest_point]
                            
   #find the offset that would improve this distance
            

        drive_cmd = AckermannDriveStamped()

        if(self.driving_forward):
            drive_cmd.drive.speed = 1.0
            #steering_wheel_angle = math.atan(2*self.wheelbase_length*math.sin(angle_error)/self.lookahead)
            drive_cmd.drive.steering_angle = self.relative_y
        else:
            drive_cmd.drive.speed = -1.0
            #steering_wheel_angle = math.atan(2*self.wheelbase_length*math.sin(angle_error)/self.lookahead)
            drive_cmd.drive.steering_angle = -self.relative_y



        #if close to goal point:
        if index_closest_point == self.trajxy_robot_frame.shape[0] - 1 and closest_dist < 0.3:
            drive_cmd.drive.speed = 0.0
        elif index_closest_point in self.checkpoint_indexes and closest_dist < 0.3:
            self.timer_seconds_remaining = 5
            self.checkpoint_indexes.pop(0)

        if self.timer_seconds_remaining > 0:
            drive_cmd.drive.speed = 0.0

        self.drive_pub.publish(drive_cmd)

        
        x = np.array((0, self.relative_x))
        y = np.array((0, self.relative_y))
        plot_line(x, y, self.path_pub, frame="/base_link")


    def store_laser_scan(self, laser_scan):
        self.laser_ranges = np.array(laser_scan.ranges)
        self.theta_vals = np.linspace(laser_scan.angle_min, laser_scan.angle_max, self.laser_ranges.shape[0])

        min_angle_check = -np.pi/8
        max_angle_check = np.pi/8

        min_angle_slice_index = int((min_angle_check - laser_scan.angle_min)/laser_scan.angle_increment)
        max_angle_slice_index = int((max_angle_check - laser_scan.angle_min)/laser_scan.angle_increment)
        
        min_range_in_front = min(laser_scan.ranges[min_angle_slice_index:max_angle_slice_index])

        if min_range_in_front < self.switch_to_drive_back_thresh:
            self.driving_forward = False
        elif min_range_in_front > self.switch_to_drive_frwrd_thresh:
            self.driving_forward = True

        self.laserx, self.lasery = self.polar_to_cartesian(self.laser_ranges, self.theta_vals)
        #self.get_logger().info("running laser callback")


    def trajectory_callback(self, msg):
        #self.get_logger().info(f"Receiving new trajectory {msg} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        x_values = [pose.position.x for pose in msg.poses]
        y_values = [pose.position.y for pose in msg.poses]

        # Combining x and y values into a single NumPy array
        self.traj_xy = np.array([x_values, y_values]).T
        #self.get_logger().info(f"running traj {str(self.traj_xy)}")

        self.initialized_traj = True

    def checkpoints_callback(self, msg):
        self.checkpoint_indexes = []
        self.checkpoint_poses = msg.poses

    def global_to_local(self, points_global, x, y, theta):
        # Translate points to the origin of the local frame
        points_local = points_global - np.array([x, y])

        # Rotate points to align with the local frame
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        points_local_rotated = points_local @ rotation_matrix

        return points_local_rotated
    
    def quaternion_to_yaw(self, x, y, z, w):
        # Computing yaw angle (rotation around z-axis)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        
        return yaw
    
    def polar_to_cartesian(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()


def plot_line(x, y, publisher, color = (1., 0., 0.), frame = "/base_link"):
    """
    Publishes the points (x, y) to publisher
    so they can be visualized in rviz as
    connected line segments.
    Args:
        x, y: The x and y values. These arrays
        must be of the same length.
        publisher: the publisher to publish to. The
        publisher must be of type Marker from the
        visualization_msgs.msg class.
        color: the RGB color of the plot.
        frame: the transformation frame to plot in.
    """
    # Construct a line
    line_strip = Marker()
    line_strip.type = Marker.LINE_STRIP
    line_strip.header.frame_id = frame

    # Set the size and color
    line_strip.scale.x = 0.1
    line_strip.scale.y = 0.1
    line_strip.color.a = 1.
    line_strip.color.r = color[0]
    line_strip.color.g = color[1]
    line_strip.color.g = color[2]

    # Fill the line with the desired values
    for xi, yi in zip(x, y):
        p = Point()
        p.x = xi
        p.y = yi
        line_strip.points.append(p)

    # Publish the line
    publisher.publish(line_strip)