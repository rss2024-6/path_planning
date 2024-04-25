import rclpy
import numpy as np

from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import distance

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 3  # FILL IN #
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

        self.laser_scan = LaserScan()
        self.laserx = np.zeros(1)
        self.lasery = np.zeros(1)

        self.relative_x = 1.
        self.relative_y = 0.
        self.min_velocity = 0.9
        self.max_velocity = 1.0
        self.current_velocity = 0.
        self.Num_points_look_ahead = 7
        self.last_solution = np.ones(2*self.Num_points_look_ahead)*0.1
        self.L_car_wheelbase = 0.5
        self.time_step = 0.3
        self.optimal_controls = np.zeros(2*self.Num_points_look_ahead)
        self.optimal_states = np.zeros([4,self.Num_points_look_ahead])
        self.max_steer = 0.34
        self.max_accel = 2
        self.min_dist_waypoint_to_lidar = 0.5
        self.safety_stop_waypoint_dist = 0.25
        self.last_path_compute_time = 0

        self.current_robot_x = 0
        self.current_robot_y = 0
        self.current_robot_theta = 0

    def store_laser_scan(self, laser_scan):
        self.laser_ranges = np.array(laser_scan.ranges)
        self.theta_vals = np.linspace(laser_scan.angle_min, laser_scan.angle_max, self.laser_ranges.shape[0])
        self.laserx, self.lasery = self.polar_to_cartesian(self.laser_ranges, self.theta_vals)
        #self.get_logger().info("running laser callback")

    def polar_to_cartesian(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

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

        self.relative_x, self.relative_y = self.trajxy_robot_frame[-1,:]

        # self.get_logger().info(f"running pose callback {str(self.relative_x), self.relative_y}")

        self.relative_cone_callback()

    def global_to_local(self, points_global, x, y, theta):
        # Translate points to the origin of the local frame
        points_local = points_global - np.array([x, y])

        # Rotate points to align with the local frame
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        points_local_rotated = points_local @ rotation_matrix

        return points_local_rotated

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {msg} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        x_values = [pose.position.x for pose in msg.poses]
        y_values = [pose.position.y for pose in msg.poses]

        # Combining x and y values into a single NumPy array
        self.traj_xy = np.array([x_values, y_values]).T
        self.get_logger().info(f"running traj {str(self.traj_xy)}")

        self.initialized_traj = True

    ###MPC STUFF BELOW###

    def recompute_optimal_controls(self):
        time_step = self.time_step
        current_state = np.array([0,0,0,self.current_velocity])

        #self.relative_x, self.relative_y = self.trajxy_robot_frame[-1]

        for i in range(self.trajxy_robot_frame.shape[0] - 1, 0, -1):
            if abs(self.trajxy_robot_frame[i, 0]) < self.lookahead and abs(self.trajxy_robot_frame[i, 1]) < self.lookahead:
                self.relative_x, self.relative_y = self.trajxy_robot_frame[i]
                break



        self.optimal_controls = self.find_mpc_control(self.relative_x, self.relative_y, self.current_velocity, time_step, controls0=self.last_solution)

        optimal_turn_angles = self.optimal_controls[:self.Num_points_look_ahead]
        optimal_accelerations = self.optimal_controls[self.Num_points_look_ahead:]

        self.optimal_states = self.make_states_from_inputs(current_state, optimal_turn_angles, optimal_accelerations, time_step)

        #display the projected path
        x = self.optimal_states[0,:]
        y = self.optimal_states[1,:]

        #x = np.array((0, self.relative_x))
        #y = np.array((0, self.relative_y))

        plot_line(x, y, self.path_pub, frame="/base_link")
    
    def min_distance(self, points1, points2):
        """Finds the minimum distance between two sets of points.

        Args:
            points1: A list of (x, y) coordinates.
            points2: A list of (x, y) coordinates.

        Returns:
            The minimum distance between any two points in the two sets.
        """

        # Convert the points to NumPy arrays.
        points1 = np.array(points1)
        points2 = np.array(points2)

        # Calculate the distance matrix.
        distance_matrix = distance.cdist(points1, points2)

        # Find the minimum distance.
        min_distance = np.min(np.abs(distance_matrix))

        return min_distance


    def relative_cone_callback(self):
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        #################################

        self.recompute_optimal_controls()
        now = self.get_clock().now().to_msg()
        current_time = int((now.sec * 1000) + (now.nanosec / 1e6))
        time_since_update = current_time - self.last_path_compute_time
        time_since_update = min(time_since_update, 200)

        self.last_path_compute_time = current_time

        optimal_turn_angles = self.optimal_controls[:self.Num_points_look_ahead]
        optimal_accelerations = self.optimal_controls[self.Num_points_look_ahead:]

        drive_cmd.drive.speed = self.current_velocity + optimal_accelerations[0]*time_since_update/1000.0
        drive_cmd.drive.speed = min(max(drive_cmd.drive.speed, self.min_velocity), self.max_velocity)
        #self.optimal_states[3,time_index + 1]
        drive_cmd.drive.steering_angle = optimal_turn_angles[0]

        time_lookahead = 1
        n_lookahead = int(time_lookahead / self.time_step)
        set1 = self.optimal_states[0:2, 0:n_lookahead]
        set1 = set1.T

        set2 = np.column_stack((self.laserx, self.lasery))

        #min_dist = self.min_distance_between_sets(set1, set2)
        min_dist = self.min_distance(set1, set2)


        if(min_dist <= self.safety_stop_waypoint_dist):
            drive_cmd.drive.speed = 0.0

        self.drive_pub.publish(drive_cmd)

    def calc_next_state(self, state, turn_angle, a, time_step):
        x, y, theta, v = state
        next_x = x + v*np.cos(theta) * time_step
        next_y = y + v*np.sin(theta) * time_step
        next_theta = theta + v*np.tan(turn_angle)/self.L_car_wheelbase * time_step
        next_v = v + a*time_step
        next_v = min(max(next_v, self.min_velocity), self.max_velocity)
        return np.array([next_x, next_y, next_theta, next_v])
    
    def make_states_from_inputs(self, start_state, turn_angle_list, a_list, time_step):
        state_list = np.zeros([4,len(turn_angle_list)])
        current_state = start_state
        state_list[:,0] = start_state
        for i in range(1, len(turn_angle_list)):
            current_state = self.calc_next_state(current_state, turn_angle_list[i], a_list[i], time_step)
            state_list[:,i] = current_state

        return state_list
    
    def find_closest_point(self, points, test_point):
        # Compute Euclidean distance between test point and each point in the array
        distances = np.linalg.norm(points - test_point, axis=1)
        
        # Find the index of the point with the minimum distance
        closest_index = np.argmin(distances)
        
        # Get the minimum distance
        min_distance = distances[closest_index]
    
        return closest_index, min_distance

    
    def cost_function(self, states, target_state, turn_angles, accelerations):
        x_current = states[0]
        y_current = states[1]
        x_goal = target_state[0]
        y_goal = target_state[1]

        #cost is how good the closest point is plus how far away you are from that point

        # Euclidean distance between current and goal positions
        #distances = np.sqrt((x_current - x_goal) ** 2 + (y_current - y_goal) ** 2)
        distances = (x_current-x_goal) + (y_current - y_goal)
        #distance_cost = np.sum(np.abs(distances - self.parking_distance))
        distance_cost = np.sum(np.abs(distances))

        # Angular difference to the target direction
        theta_to_cone = np.arctan2((y_goal - y_current), (x_goal - x_current))
        angle_cost = np.abs(theta_to_cone - states[2])[-1]#np.sum(np.abs(theta_to_cone - states[2]) * 2)

        # Regularization terms for turn angles and accelerations
        turn_angle_reg = np.sum(np.abs(np.diff(turn_angles))) * 0.02
        acceleration_reg = np.sum(np.abs(np.diff(accelerations))) * 0.01

        set1 = states[0:2, :]
        set1 = set1.T
        #self.get_logger().info(str(set1.shape))
        set2 = np.column_stack((self.laserx, self.lasery))
        #self.get_logger().info(str(set2.shape))

        collision_cost = 0

        min_dist = self.min_distance(set1, set2)

        if min_dist < self.min_dist_waypoint_to_lidar:
            collision_cost = 999/max(min_dist, 0.0001)

        return distance_cost + angle_cost + turn_angle_reg + acceleration_reg + collision_cost

    def mpc_controller(self, initial_state, time_step, target_state, controls0 = 0 ):
        state = initial_state
        controls_opt = None

        # Define the optimization problem
        def objective(controls):
            turn_angles = controls[:self.Num_points_look_ahead]
            accelerations = controls[self.Num_points_look_ahead:]
            next_states = self.make_states_from_inputs(state, turn_angles, accelerations, time_step)
            return self.cost_function(next_states, target_state, turn_angles, accelerations)
        
        if type(controls0) == int:
            # Initial guess for controls
            controls0 = (np.zeros(2*self.Num_points_look_ahead))
            #controls0 = np.zeros(2*N)

        # Bounds for controls (optional but often necessary)
        bounds = [(-0.34, 0.34)] * self.Num_points_look_ahead + [(-self.max_accel, self.max_accel)] * self.Num_points_look_ahead  # Example bounds for turn angle and acceleration        
        
        lowest_score = np.inf
        for turn in np.arange(-self.max_steer, self.max_steer, (2*self.max_steer)/5):
            for accel in np.arange(-self.max_accel, self.max_accel, (2*self.max_accel)/3):
                this_controls0 = np.hstack((np.ones(self.Num_points_look_ahead)*turn, np.ones(self.Num_points_look_ahead)*accel))
                these_states = self.make_states_from_inputs(initial_state, this_controls0[:self.Num_points_look_ahead], this_controls0[self.Num_points_look_ahead:], time_step= self.time_step)
                this_cost = self.cost_function(these_states, target_state, this_controls0[:self.Num_points_look_ahead], this_controls0[self.Num_points_look_ahead:])
                if this_cost < lowest_score:
                    lowest_score = this_cost
                    controls_opt = this_controls0
        

        max_iterations = 5
        result = minimize(objective, controls_opt, bounds=bounds, options={'maxiter': max_iterations}) #constraints=acceleration_constraint , bounds=bounds , method='L-BFGS-B'
        controls_opt = result.x

        return controls_opt
    
    def find_mpc_control(self, cone_x, cone_y, current_velocity, time_step, controls0 = 0):
        current_state = np.array([0,0,0,current_velocity])
        target_state = np.array([cone_x, cone_y, 0, 0])

        optimal_controls = self.mpc_controller(current_state, time_step, target_state, controls0=controls0)
        return optimal_controls
    
    def quaternion_to_yaw(self, x, y, z, w):
        # Computing yaw angle (rotation around z-axis)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        
        return yaw



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