import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Point, PointStamped, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from .utils import LineTrajectory
import numpy as np
import tf_transformations
import scipy
import skimage.morphology as ski
import cv2
import math
import heapq
import json
import os
import time

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
        super().__init__("final_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map = OccupancyGrid()

        #A* Params
        self.downsampling_factor = 7
        self.real_time_viz = False
        self.lane_offset = 0.2
        self.neighbor_options = "cardinal" # "knight" or "cardinal"
        self.grid_viz = "lanes-dirs" #none or "full", "full-dirs", "lanes", "lanes-dirs"

        # A* Variables
        self.ROW = None
        self.COL = None
        self.DOWNSAMPLED_ROW = None
        self.DOWNSAMPLED_COL = None
        self.pos = [0, 0]
        self.map_data = np.array([0,0])
        self.downsampled_map = None
        self.current_start_point = [0,0]


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

        self.clicked_point_sub = self.create_subscription(PointStamped, "/clicked_point", self.clicked_point_cb, 10)

        # Visualizations
        self.closest_line_pub = self.create_publisher(Marker, "/closest_line", 10)
        self.lane_line_directions_pub = self.create_publisher(PoseArray, "/lane_line_directions", 10)
        self.cardinal_dir_pub = self.create_publisher(PoseStamped, "/cardinal_direction", 10)
        self.lanes_pub = self.create_publisher(Marker,"/lanes",10)
        self.lanes_timer = self.create_timer(1, self.visualize_lanes)
        self.visited_pub = self.create_publisher(Marker, "/visited", 10)
        self.visited_dir_pub = self.create_publisher(PoseArray, "/visited_directions", 10)
        self.grid_pub = self.create_publisher(Marker, "/grid", 10)
        self.grid_dir_pub = self.create_publisher(PoseArray, "/grid_directions", 10)
        self.selected_cell_pub = self.create_publisher(Marker, "/selected_cell", 10) 
        self.selected_dir_pub = self.create_publisher(Marker, "/selected_direction", 10) 
        self.valid_neighbor_dir_pub = self.create_publisher(PoseArray, "/valid_neighbor_directions", 10)
        self.checkpoints_viz_pub = self.create_publisher(Marker, "/checkpoints_viz", 10)

        #Trajectory
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.checkpoints = []
        self.checkpoints_pub = self.create_publisher(PoseArray, "/checkpoints", 10)

        # Load lane points
        with open('src/path_planning/lanes/full-lane.traj', 'r') as file:
            data = json.load(file)

        # Define the points
        data_points = data['points']
        self.lane_points = [Point(x=dp['x'], y=dp['y']) for dp in data_points]

        self.visited = []

        self.get_logger().info("---------------READY---------------")


    def map_cb(self, msg):
        """
        Runs when map is received
        """
        self.map = msg
        self.COL = msg.info.width
        self.ROW = msg.info.height

        data = np.array(msg.data).reshape((self.ROW,self.COL))
        blurred_data = ski.dilation(data, ski.square(7))
        cv2.imwrite('/home/racecar/racecar_ws/res_15.png',blurred_data)
        self.get_logger().info('Map Found!')
        self.map_data = blurred_data
        self.remove_top_cells(self.map_data)
        self.downsampled_map = self.map_data[::self.downsampling_factor,::self.downsampling_factor]
        self.DOWNSAMPLED_ROW = self.downsampled_map.shape[0]
        self.DOWNSAMPLED_COL = self.downsampled_map.shape[1]

        if self.grid_viz != "none":
            lane_only = "lanes" in self.grid_viz
            directions = "dirs" in self.grid_viz
            self.visualize_grid(self.downsampled_map, show_directions=directions, lane_only=lane_only)

        # self.get_logger().info("map shape %s" % str(self.map_data.shape))
        # self.get_logger().info("ROW %d" % self.ROW)
        # self.get_logger().info("COL %d" % self.COL)
        # self.get_logger().info("ds map shape %s" % str(self.downsampled_map.shape))
        # self.get_logger().info("DOWNSAMPLED_ROW %d" % self.DOWNSAMPLED_ROW)
        # self.get_logger().info("DOWNSAMPLED_ROW %d" % self.DOWNSAMPLED_COL)
    
    def remove_top_cells(self, grid):
        """
        Turns the rightmost pixels of the map into obstacles
        """
        grid[800:,1000:1500] = 100
        # grid[:,:] = 100 
        


    def pose_cb(self, pose):
        """
        Runs when start point is received through '2D Pose Estimate'
        """
        self.trajectory.clear()
        self.pos = [pose.pose.pose.position.x, pose.pose.pose.position.y]
        self.current_start_point = self.pos
        self.checkpoints = []
        self.visualize_checkpoints()

    def goal_cb(self, msg):
        """
        Runs when goal point is received through '2D Goal Pose'
        """
        # self.test_card_directions(msg.pose)
        # self.test_valid_neighbors(msg.pose)
        # return 
        self.checkpoints.append(msg.pose)

        self.plan_path(self.current_start_point, [msg.pose.position.x, msg.pose.position.y])

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        pose_array.poses = self.checkpoints
        self.checkpoints_pub.publish(pose_array)
        self.traj_pub.publish(self.trajectory.toPoseArray())

        self.trajectory.publish_viz()

    
    def clicked_point_cb(self, msg):
        """
        Runs when point is added to agenda through 'Publish Point'
        """
        # self.test_lane_classification(msg)
        # return 
        pose = Pose()
        pose.position = msg.point

        # Set default orientation (no rotation)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    
        self.checkpoints.append(pose)
        self.visualize_checkpoints()

        point = [msg.point.x, msg.point.y]
        self.plan_path(self.current_start_point, point)
        self.current_start_point = point

        # self.checkpoints.append(pose)
        # self.visualize_checkpoints()

    def plan_path(self, start_point, end_point):
        """
        Calls A star to plan path and adds result to self.trajectory
        """
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

            # self.traj_pub.publish(self.trajectory.toPoseArray())
            # self.trajectory.publish_viz()

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
    
    # Calculate the heuristic value of a cell (Diagonal distance to destination)
    def calculate_h_value(self, row, col, dest):
        D = 1
        D2 = (2) ** 0.5
        dx = abs(row - dest[0])
        dy = abs(col - dest[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    
    # Trace the path from source to destination
    def trace_path(self, cell_details, dest):
        """
        Traces the path from source to destination
        """
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
        self.get_logger().info("A* Search Started")
        self.visited = []
        # Check if the source and destination are valid
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            self.get_logger().info("Source or destination is invalid")
            return
    
        # Check if the source and destination are unblocked
        if not self.is_unblocked(grid, src[0], src[1]) or not self.is_unblocked(grid, dest[0], dest[1]):
            self.get_logger().info("Source or the destination is blocked")
            return
    
        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            self.get_logger().info("We are already at the destination")
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
            self.visited.append((i, j))
            if self.real_time_viz: self.visualize_visited(show_directions=True)
    
            # For each direction, check the successors
            directions = self.get_neighbor_directions((i, j))
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]
                # If the successor is valid, unblocked, and not visited
                if self.is_valid(new_i, new_j) and self.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j] and self.is_valid_neighbor((new_i, new_j), dir):
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
    
    
    def get_neighbor_directions(self, cell):
        """
        Gives possible directions to move from a given cell taking into account lane direction constraints
        """

        #Cyclically ordered cardinal directions
        if self.neighbor_options == "cardinal":
            ordered_directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        elif self.neighbor_options == "knight":
            ordered_directions = [(0, 1), (-1,2), (-1, 1), (-2, 1),(-1, 0), (-2,-1), (-1, -1), (-1, -2),(0, -1), (1, -2), (1, -1), (2,-1), (1, 0), (2,1), (1, 1), (1,2)]
        
        #Index offsets for calculating range of usable directions will get directions from [start_offset, end_offset] (inclusive)
        start_offset = -2 if self.neighbor_options == "cardinal" else -4
        end_offset = 2 if self.neighbor_options == "cardinal" else 4

        #Get closest lane line
        map_point = self.pixel_to_map(cell[0], cell[1])
        map_point = Point(x=map_point[0], y=map_point[1])
        closest_line = self.find_closest_line(map_point)

        #Get cardinal lane direction
        lane_direction = self.get_lane_direction(cell)

        #Get all directions that are ahead or next to the lane direction
        dir_index = ordered_directions.index(lane_direction)


        # Handle points that are too close to lane line
        if self.point_line_distance(map_point.x, map_point.y, closest_line[0].x, closest_line[0].y, closest_line[1].x, closest_line[1].y) < self.lane_offset:
            #Get all directions that go forwards 
            # forward_diag = [ordered_directions[(dir_index + i) % 8] for i in range(start_offset, end_offset)]
            # #Remove the straight forward direction
            # forward_diag.remove(ordered_directions[dir_index])

            if self.neighbor_options == "cardinal":
                just_sides = [ordered_directions[(dir_index + -2) % 8], ordered_directions[(dir_index + 2) % 8]]
            else:
                just_sides = [ordered_directions[(dir_index + -4) % 16], ordered_directions[(dir_index + 4) % 16]]
            return just_sides

        # Now handle all other points
        if self.neighbor_options == "cardinal":
            return [ordered_directions[(dir_index + i) % 8] for i in range(start_offset, end_offset+1)]
        else:
            return [ordered_directions[(dir_index + i) % 16] for i in range(start_offset, end_offset+1)]
    
    def is_valid_neighbor(self, cell, dir):
        """
        Final check to see if neighbor is valid. If the direction moved to get to the neighbor is 
        in the opposite direction of the neighbors lane direction then the neighbor is invalid

        Returns True if the neighbor is valid
        """
        map_point = self.pixel_to_map(cell[0], cell[1])
        map_point = Point(x=map_point[0], y=map_point[1])
        #Get cardinal direction most parallel to the lane line
        lane_direction = self.get_lane_direction(cell)
        unit_ld = np.array([lane_direction[0], lane_direction[1]])
        unit_ld = unit_ld / np.linalg.norm(unit_ld)

        unit_dir = np.array([dir[0], dir[1]])
        unit_dir = unit_dir / np.linalg.norm(unit_dir)

        dot_product = np.dot(unit_dir, unit_ld)
        condition = dot_product >= 0

        # self.get_logger().info(f'Dir:{dir}')
        # self.get_logger().info(f'LD:{lane_direction}')
        # self.get_logger().info(f'Dot Product: {dot_product}')

        return condition
    
    def get_lane_direction(self, cell):
        """
        Returns the lane direction of the cell
        """
        map_point = self.pixel_to_map(cell[0], cell[1])
        map_point = Point(x=map_point[0], y=map_point[1])
        closest_line = self.find_closest_line(map_point)
        #Get cardinal direction most parallel to the lane line
        lane_direction = self.get_cardinal_direction(closest_line)
        if self.in_left_lane(map_point):
            lane_direction = (-lane_direction[0], -lane_direction[1])
        return lane_direction


    def get_cardinal_direction(self, line):
        """
        Returns one of the cardinal directions [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        that is most parallel to the given line
        """
        if self.neighbor_options == "cardinal":
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            directions = [(0, 1), (-1,2), (-1, 1), (-2, 1),(-1, 0), (-2,-1), (-1, -1), (-1, -2),(0, -1), (1, -2), (1, -1), (2,-1), (1, 0), (2,1), (1, 1), (1,2)]

        unit_directions = []
        for direction in directions:
            vector = np.array([direction[0], direction[1]])
            vector = vector / np.linalg.norm(vector)
            unit_directions.append(vector)
        
        # convert map coords to pixel to get lines unit vector
        p1 = np.array(self.map_to_pixel(np.array([line[0].x, line[0].y])), dtype=np.float64)
        p2 = np.array(self.map_to_pixel(np.array([line[1].x, line[1].y])), dtype=np.float64)

        unit_vector = p1-p2
        unit_vector /= np.linalg.norm(unit_vector)
        min_dot = np.inf
        most_parallel_direction = None
        
        for index, ud in enumerate(unit_directions):
            dot_product = np.dot(unit_vector, ud)
            if dot_product < min_dot:
                min_dot = dot_product
                most_parallel_direction = directions[index]

        return most_parallel_direction

    def in_left_lane(self, point):
        lane_line = self.find_closest_line(point)
        lane_vec = np.array([lane_line[0].x - lane_line[1].x, lane_line[0].y - lane_line[1].y])
        point_vec = np.array([point.x - lane_line[0].x, point.y - lane_line[0].y])

        # Calculate the 2D cross product
        cross_product = np.cross(lane_vec, point_vec)

        return cross_product < 0

    def point_line_distance(self, px, py, x1, y1, x2, y2):
        # Line coefficients
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1

        # Perpendicular distance
        dist = abs(A * px + B * py - C) / math.sqrt(A**2 + B**2)
        # Projection point on the line
        dx = x2 - x1
        dy = y2 - y1
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        if t < 0.0 or t > 1.0:  # Outside the segment, use the closest endpoint
            if t < 0.0:
                dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            else:
                dist = math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
        return dist
    
    def find_closest_line(self, point):
        """
        Returns closest lane line to a point in the map space
        """
        min_distance = float('inf')
        closest_segment = None

        # Iterate over pairs of points to define line segments
        for i in range(len(self.lane_points) - 1):
            p1 = self.lane_points[i]
            p2 = self.lane_points[i + 1]
            distance = self.point_line_distance(point.x, point.y, p1.x, p1.y, p2.x, p2.y)
            if distance < min_distance:
                min_distance = distance
                closest_segment = (p1, p2)
        return closest_segment

    # Visualizations

    def visualize_checkpoints(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "spheres"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0  # Sphere diameter
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        for checkpoint in self.checkpoints:  # Example with 5 spheres
            point = checkpoint.position
            color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            marker.points.append(point)
            marker.colors.append(color)

        # marker.lifetime = rcslpy.duration.Duration()

        self.checkpoints_viz_pub.publish(marker)


        # self.checkpoints_viz_pub.publish(marker_array)
    def visualize_lanes(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lines"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        with open('src/path_planning/lanes/full-lane.traj', 'r') as file:
            # Load its content and convert it into a dictionary
            data = json.load(file)

        # Define the points
        data_points = data['points']
        points = [Point(x=dp['x'], y=dp['y']) for dp in data_points]

        marker.points = points

        # Define the line color and scale
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        self.lanes_pub.publish(marker)
        # self.show_lane_directions()
    
    
    def test_lane_classification(self, msg):
        point = msg.point

        in_left_lane = self.in_left_lane(point)
        closest_line = self.find_closest_line(point)

        self.get_logger().info(f'Point: ({point.x}, {point.y})')
        self.get_logger().info(f'In Left Lane: {in_left_lane}')
        self.get_logger().info(f'Closest Line: ({closest_line[0].x}, {closest_line[0].y}), ({closest_line[1].x}, {closest_line[1].y})')

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lines"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        points = closest_line
        marker.points = points

        # Define the line color and scale
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0  # Alpha
        marker.color.r = 0.0  # Red
        marker.color.g = 1.0  # Green
        marker.color.b = 0.0  # Blue

        self.closest_line_pub.publish(marker)

    def show_lane_directions(self):
        """
        For debugging shows directions of lane lines
        """       
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        for i in range(len(self.lane_points) - 1):
            p1 = self.lane_points[i]
            p2 = self.lane_points[i + 1]
            direction = self.get_cardinal_direction([p1, p2])
            yaw_angle = math.atan2(direction[1], direction[0])
            quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
            pose1 = Pose()
            pose1.position = p1
            pose1.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])

            pose_array.poses.append(pose1)
            # pose_array.poses.append(pose2)
        self.lane_line_directions_pub.publish(pose_array)
    
    def test_card_directions(self, pose):
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        roll, pitch, yaw = tf_transformations.euler_from_quaternion(orientation)

        dx, dy = math.cos(yaw)*2, math.sin(yaw)*2
        start = Point(x=position[0], y=position[1])
        end = Point(x=position[0] + dx, y=position[1] + dy)

        direction = self.get_cardinal_direction([start, end])
        self.get_logger().info(f'Direction: {direction}')

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        cardinal_pose = Pose()
        cardinal_pose.position = start
        org = self.pixel_to_map(0, 0)
        dir_pos = self.pixel_to_map(direction[0], direction[1])
        yaw_angle = math.atan2(dir_pos[1] - org[1], dir_pos[0] - org[0])
        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
        cardinal_pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        msg.pose = cardinal_pose

        self.cardinal_dir_pub.publish(msg)
    
    def test_valid_neighbors(self, pose):
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(orientation)
        dx, dy = math.cos(yaw)*2, math.sin(yaw)*2
        start = Point(x=position[0], y=position[1])
        end = Point(x=position[0] + dx, y=position[1] + dy)
        direction = self.get_cardinal_direction([start, end])

        cell_r, cell_c = self.map_to_pixel([position[0], position[1]])
        n_r = cell_r + direction[0]
        n_c = cell_c + direction[1]

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "visited"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        cell_map_point = self.pixel_to_map(cell_r, cell_c)
        neighbor_map_point = self.pixel_to_map(n_r, n_c)
        marker.points = [Point(x=cell_map_point[0], y=cell_map_point[1]), Point(x=neighbor_map_point[0], y=neighbor_map_point[1])]

        marker.scale.x = .1
        marker.scale.y = .1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        if self.is_valid_neighbor((n_r, n_c), direction):
            self.get_logger().info("Valid Neighbor")
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = "map"
            pose = Pose()
            pose.position = Point(x=cell_map_point[0], y=cell_map_point[1])
            org = self.pixel_to_map(0, 0)
            dir_pos = self.pixel_to_map(direction[0], direction[1])
            yaw_angle = math.atan2(dir_pos[1] - org[1], dir_pos[0] - org[0])
            quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
            pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
            pose_array.poses.append(pose)
            self.valid_neighbor_dir_pub.publish(pose_array)
        else:
            self.get_logger().info("Invalid Neighbor")

        self.selected_cell_pub.publish(marker)


    def visualize_visited(self, show_directions=False):
        self.get_logger().info("Visualizing visited cells")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "visited"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        points = []
        for cell in self.visited:
            np_point = self.pixel_to_map(cell[0], cell[1])
            point = Point(x=np_point[0], y=np_point[1])
            points.append(point)

            if show_directions:
                for dir in self.get_neighbor_directions(cell):
                    new_i = cell[0] + dir[0]
                    new_j = cell[1] + dir[1]
                    if self.is_valid_neighbor((new_i, new_j), dir):
                        pose = Pose()
                        pose.position = point
                        org = self.pixel_to_map(0, 0)
                        dir_pos = self.pixel_to_map(dir[0], dir[1])
                        yaw_angle = math.atan2(dir_pos[1] - org[1], dir_pos[0] - org[0])
                        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
                        pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
                        pose_array.poses.append(pose)

        marker.points = points

        # Define the line color and scale
        marker.scale.x = .1
        marker.scale.y = .1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        self.visited_pub.publish(marker)
        self.visited_dir_pub.publish(pose_array)
    
    def visualize_grid(self, grid, show_directions=False, lane_only = False):
        self.get_logger().info("Visualizing grid")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "grid"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        points = []
        for i in range(self.DOWNSAMPLED_ROW):
            for j in range(self.DOWNSAMPLED_COL):
                if grid[i][j] == 0:
                    np_point = self.pixel_to_map(i, j)
                    point = Point(x=np_point[0], y=np_point[1])

                    if lane_only:
                        closest_line = self.find_closest_line(point)
                        if self.point_line_distance(point.x, point.y, closest_line[0].x, closest_line[0].y, closest_line[1].x, closest_line[1].y) < self.lane_offset:
                            points.append(point)
                    else:
                        points.append(point)
                    
                    if show_directions:
                        lane_direction = self.get_lane_direction((i, j))
                        pose = Pose()
                        pose.position = point
                        org = self.pixel_to_map(0, 0)
                        dir_pos = self.pixel_to_map(lane_direction[0], lane_direction[1])
                        yaw_angle = math.atan2(dir_pos[1] - org[1], dir_pos[0] - org[0])
                        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)
                        pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
                        pose_array.poses.append(pose)
        
        marker.points = points

        # Define the line color and scale
        marker.scale.x = .1
        marker.scale.y = .1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.grid_pub.publish(marker)
        self.grid_dir_pub.publish(pose_array)










def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
