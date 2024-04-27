import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import tf_transformations
import scipy
import skimage.morphology as ski
import cv2
import math
import heapq

# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

class PathPlan(Node):
    """
    A* Path Planning Algorithm 
    Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map = OccupancyGrid()

        #A* Params
        self.downsampling_factor = 15

        # A* Variables
        self.ROW = None
        self.COL = None
        self.DOWNSAMPLED_ROW = None
        self.DOWNSAMPLED_COL = None
        self.pos = [0, 0]
        self.map_data = np.array([0,0])
        self.downsampled_map = None


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
        self.map = msg
        self.COL = msg.info.width
        self.ROW = msg.info.height

        data = np.array(msg.data).reshape((self.ROW,self.COL))
        blurred_data = ski.dilation(data, ski.square(15))
        cv2.imwrite('/home/racecar/racecar_ws/res_15.png',blurred_data)
        self.get_logger().info('Map Found!')
        self.map_data = blurred_data
        self.downsampled_map = self.map_data[::self.downsampling_factor,::self.downsampling_factor]
        self.DOWNSAMPLED_ROW = self.downsampled_map.shape[0]
        self.DOWNSAMPLED_COL = self.downsampled_map.shape[1]

        # self.get_logger().info("map shape %s" % str(self.map_data.shape))
        # self.get_logger().info("ROW %d" % self.ROW)
        # self.get_logger().info("COL %d" % self.COL)
        # self.get_logger().info("ds map shape %s" % str(self.downsampled_map.shape))
        # self.get_logger().info("DOWNSAMPLED_ROW %d" % self.DOWNSAMPLED_ROW)
        # self.get_logger().info("DOWNSAMPLED_ROW %d" % self.DOWNSAMPLED_COL)
        


    def pose_cb(self, pose):
        self.pos = [pose.pose.pose.position.x, pose.pose.pose.position.y]

    def goal_cb(self, msg):
        self.trajectory.clear()
        self.plan_path(self.pos, [msg.pose.position.x, msg.pose.position.y], self.map, self.map_data)

    def plan_path(self, start_point, end_point, map, mapdata):
        # print("downsampled map",self.downsampled_map)
        # self.get_logger().info("start %s" % start_point)
        # self.get_logger().info("end %s" % end_point)
        t_start = self.get_clock().now().nanoseconds / 1e9        # BEGIN ALGORITHM

        start = self.map_to_pixel(start_point)
        end = self.map_to_pixel(end_point)
        
        path = self.a_star_search(self.downsampled_map, start, end)

        if path is not None:
            self.get_logger().info('Path found!')
            #add path to trajectory
            for point in path:
                self.trajectory.addPoint(self.pixel_to_map(point[0], point[1]))

            t_end = self.get_clock().now().nanoseconds / 1e9        # END ALGORITHM
            Q_MET = (self.trajectory.distances[-1])/(t_end-t_start)

            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()

            self.get_logger().info('Path Distance: %s' % str(self.trajectory.distances[-1]))
            self.get_logger().info('Runtime: %s' % str(t_end-t_start))
            self.get_logger().info('Efficiency: %s' % str(Q_MET))

        else:
            self.get_logger().info('No path found')

        

    # Check if a cell is valid (within the grid)
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.DOWNSAMPLED_ROW) and (col >= 0) and (col < self.DOWNSAMPLED_COL)

    # Check if a cell is unblocked
    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 0
    
    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]
    
    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        D = 1
        D2 = (2) ** 0.5
        dx = abs(row - dest[0])
        dy = abs(col - dest[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    # Trace the path from source to destination
    def trace_path(self, cell_details, dest):
        print("The Path is ")
        path = []
        row = dest[0]
        col = dest[1]
    
        # Trace the path from destination to source using parent cells
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col
    
        # Add the source cell to the path
        path.append((row, col))
        # Reverse the path to get the path from source to destination
        path.reverse()

        return path
    
    # Implement the A* search algorithm
    def a_star_search(self, grid, src, dest):
        # Check if the source and destination are valid
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return
    
        # Check if the source and destination are unblocked
        if not self.is_unblocked(grid, src[0], src[1]) or not self.is_unblocked(grid, dest[0], dest[1]):
            print("Source or the destination is blocked")
            return
    
        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return
    
        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.DOWNSAMPLED_COL)] for _ in range(self.DOWNSAMPLED_ROW)]
        # Initialize the details of each cell
        cell_details = [[Cell() for _ in range(self.DOWNSAMPLED_COL)] for _ in range(self.DOWNSAMPLED_ROW)]
    
        # Initialize the start cell details
        i = src[0]
        j = src[1]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j
    
        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
    
        # Initialize the flag for whether destination is found
        found_dest = False
    
        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)
    
            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True
    
            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]
    
                # If the successor is valid, unblocked, and not visited
                if self.is_valid(new_i, new_j) and self.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if self.is_destination(new_i, new_j, dest):
                        # Set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        return self.trace_path(cell_details, dest)
                        # found_dest = True
                        # return
                    else:
                        # Calculate the new f, g, and h values
                        # TODO: Change g_new
                        g_new = cell_details[i][j].g + math.sqrt((i - new_i) ** 2 + (j - new_j) ** 2)
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new
    
                        # If the cell is not in the open list or the new f value is smaller
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j
    
        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")
        


      

    def pixel_to_map(self, r, c):
        upsampled_r = r * self.downsampling_factor
        upsampled_c = c * self.downsampling_factor

        scaled_r = upsampled_r * self.map.info.resolution
        scaled_c = upsampled_c * self.map.info.resolution

        pose = self.map.info.origin

        # Extract the translation and rotation from map
        translation = np.array([pose.position.x, pose.position.y, pose.position.z]).T
        rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert the quaternion to a rotation matrix
        T = tf_transformations.quaternion_matrix(rotation)
        T[:3, 3] = translation

        # Create a 4x1 homogeneous matrix for the point
        point_homogeneous = np.array([scaled_c, scaled_r, 0, 1]).T

        # Apply the rotation and translation
        transformed_point = np.dot(T, point_homogeneous)

        # Extract the x, y, z coordinates (ignore the homogeneous coordinate)
        x_transformed, y_transformed, _ = transformed_point[:3]
        return np.array([x_transformed, y_transformed])
        
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

        downsampled_r = scaled_y / self.downsampling_factor
        downsampled_c = scaled_x / self.downsampling_factor

        return (int(downsampled_r), int(downsampled_c))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
