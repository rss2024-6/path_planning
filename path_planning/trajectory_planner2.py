import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Point
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
from scipy.spatial import KDTree
import tf_transformations as tfm
import scipy
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import skimage.morphology as ski
import cv2
import tf_transformations 
import time

class TreeNode:
    def __init__(self, position):
        self.position = position  # position now includes x, y, theta
        self.parent = None
        self.cost = 0   

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner2")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map = OccupancyGrid()
        self.pos = [0, 0]

        # RRT* Params
        self.debug = False
        self.visualize_tree = True
        self.iterations = 1000
        self.goal_sample_rate = 0.1
        self.step_length = .1
        self.search_radius = 0.5
        

        # RRT* Variables
        self.tree = None
        self.node_list = []
        self.start_point = None
        self.end_point = None
        self.width = None
        self.height = None
        self.occupied = None

        self.tree_publisher = self.create_publisher(Marker, 'tree_visualization', 10)
        self.goal_publisher = self.create_publisher(Marker, 'goal_visualization', 10)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg):
        # self.map = msg

        self.map = msg
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data).reshape((h,w))
        #blur = a = np.ones([10, 10], dtype = int) 
        #blurred_data = scipy.signal.convolve2d(data,blur)
        blurred_data = ski.dilation(data, ski.square(10))
        cv2.imwrite('/home/racecar/racecar_ws/res_10.png',blurred_data)
        self.get_logger().info('Map Found!')
        # self.map_data = list(blurred_data.flatten().astype('int8'))

    def pose_cb(self, pose):
        self.occupied = self.map.data
        self.width = self.map.info.width
        self.height = self.map.info.height

        self.pos = [pose.pose.pose.position.x, pose.pose.pose.position.y]

        clear = self.is_clear(TreeNode(self.pos))

        self.get_logger().info('Frame id: %s' % pose.header.frame_id)
        self.get_logger().info('Clear: %s' % clear)

    def goal_cb(self, msg):
        self.trajectory.clear()
        self.plan_path(self.pos, [msg.pose.position.x, msg.pose.position.y], self.map)

    def plan_path(self, start_point, end_point, map):
        self.occupied = map.data
        self.width = map.info.width
        self.height = map.info.height
        ori = map.info.origin.position
        res = map.info.resolution

        if self.width == 0 or self.height == 0: 
            raise ValueError("Map dimensions are 0 did you reset simulator?")

        self.start_point = (float(start_point[0]), float(start_point[1]))
        self.end_point = (float(end_point[0]), float(end_point[1]))
        self.visualize_goal()
        self.tree = KDTree([self.start_point])

        path = self.plan()

        if path is not None:
            self.get_logger().info('Path found!')
            #add path to trajectory
            for point in path:
                self.trajectory.addPoint(point)
            
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            self.get_logger().info('No path found')

        # self.publish_tree_markers(self.node_list)
        

                
    def plan(self):
        self.node_list = [TreeNode(self.start_point)]
        self.tree = KDTree([self.start_point])
        for i in range(self.iterations):
            # Sample point
            sample = self.end_point if np.random.rand() < self.goal_sample_rate else self.random_sample()

            # Get nearest node
            nearest_idx = self.tree.query([sample], 1)[1][0]
            nearest_node = self.node_list[nearest_idx]

            # Steer towards sample
            new_node = self.steer(nearest_node, sample)
            if self.is_clear(new_node):
                # Find near nodes and choose the best parent
                near_idxs = self.tree.query_ball_point(new_node.position, self.search_radius)
                new_node = self.choose_parent(new_node, near_idxs)

                if new_node:
                    self.node_list.append(new_node)
                    self.tree = KDTree([node.position for node in self.node_list]) # Rebuild tree
                    self.rewire(new_node, near_idxs)
            else:
                continue

            if self.is_near_goal(new_node):
                return self.generate_final_course(len(self.node_list) - 1)
            
            if self.visualize_tree: self.publish_tree_markers(self.node_list)
            if self.debug: self.get_logger().info('Iteration: %d' % i)
            if self.debug: time.sleep(.1)
            
        return None

    
    def steer(self, from_node, to_point):
        # Create a new node moving towards to_point from from_node
        direction = np.array(to_point) - np.array(from_node.position)
        distance = np.linalg.norm(direction)
        direction = direction / distance
        length = min(self.step_length, distance)
        new_position = np.array(from_node.position) + direction * length
        new_node = TreeNode(new_position)
        new_node.cost = from_node.cost + length
        new_node.parent = from_node
        return new_node

    def is_clear(self, node):
        # returns true if node doesn't collide with an obstacle
        x, y = node.position
        px, py = self.map_to_pixel((x, y))

        index = int(py*self.width + px)
        occupied_val = self.occupied[index]

        return occupied_val == 0 


    def choose_parent(self, new_node, near_idxs):
        # Choose the best parent for a new node (if there is a more optimal route)
        if not near_idxs:
            return None
        
        min_cost = float('inf')
        best_node = None
        for idx in near_idxs:
            node = self.node_list[idx]
            # check if it is less costly to use a neighbor node as the parent
            if self.is_clear(node) and (node.cost + np.linalg.norm(np.array(node.position) - np.array(new_node.position)) < min_cost):
                best_node = node
        if best_node:
            new_node.cost = min_cost
            new_node.parent = best_node
        return new_node
    
    def rewire(self, new_node, near_idxs):
        for idx in near_idxs:
            node = self.node_list[idx]
            if self.is_clear(node) and new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(node.position)) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + np.linalg.norm(np.array(new_node.position) - np.array(node.position))
    
    def is_near_goal(self, node):
        dx = node.position[0] - self.end_point[0]
        dy = node.position[1] - self.end_point[1]
        return dx ** 2 + dy ** 2 <= self.step_length ** 2

    def generate_final_course(self, goal_idx):
        path = [self.end_point]
        node = self.node_list[goal_idx]
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(node.position)
        return path

    
    def random_sample(self):
        num = np.random.randint(0, high = self.width*self.height)
        x = float(num % self.width)
        y = float(num // self.height)
        return self.pixel_to_map((x, y))


    def publish_tree_markers(self, nodes):
        marker = Marker()
        marker.header.frame_id = "map"  # Change to your frame ID
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Line width
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red color

        for node in nodes:
            if node.parent:
                start = Point(x=node.parent.position[0], y=node.parent.position[1], z=0.0)
                end = Point(x=node.position[0], y=node.position[1], z=0.0)
                marker.points.append(start)
                marker.points.append(end)

        self.tree_publisher.publish(marker)
    
    def pixel_to_map(self, position):
        scaled_x = position[0] * self.map.info.resolution
        scaled_y = position[1] * self.map.info.resolution

        pose = self.map.info.origin

        # Extract the translation and rotation from map
        translation = np.array([pose.position.x, pose.position.y, pose.position.z]).T
        rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert the quaternion to a rotation matrix
        T = tf_transformations.quaternion_matrix(rotation)
        T[:3, 3] = translation

        # Create a 4x1 homogeneous matrix for the point
        point_homogeneous = np.array([scaled_x, scaled_y, 0, 1]).T

        # Apply the rotation and translation
        transformed_point = np.dot(T, point_homogeneous)

        # Extract the x, y, z coordinates (ignore the homogeneous coordinate)
        x_transformed, y_transformed, _ = transformed_point[:3]
        return (x_transformed, y_transformed)
        
    def map_to_pixel(self, position):
        x = position[0]
        y = position[1]

        pose = self.map.info.origin

        # Extract the translation and rotation from map
        translation = np.array([pose.position.x, pose.position.y, pose.position.z]).T
        rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert the quaternion to a rotation matrix
        T = tf_transformations.quaternion_matrix(rotation)
        T[:3, 3] = translation

        # Create a 4x1 homogeneous matrix for the point
        point_homogeneous = np.array([x, y, 0, 1]).T

        # Apply the rotation and translation
        transformed_point = np.dot(np.linalg.inv(T), point_homogeneous)

        # Extract the x, y, z coordinates (ignore the homogeneous coordinate)
        x_transformed, y_transformed, _ = transformed_point[:3]

        # Scale coordinates to pixel space
        scaled_x = x_transformed / self.map.info.resolution
        scaled_y = y_transformed / self.map.info.resolution

        return (int(scaled_x), int(scaled_y))

    def visualize_goal(self):
        marker = Marker()
        marker.header.frame_id = "map"  # Use appropriate frame ID for your application
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = self.end_point[0]
        marker.pose.position.y = self.end_point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0  # Diameter of the sphere
        marker.scale.y = 1.0  # Diameter of the sphere
        marker.scale.z = 1.0  # Diameter of the sphere
        marker.color.a = 1.0  # Alpha, 1 means fully opaque
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        self.goal_publisher.publish(marker)





def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()