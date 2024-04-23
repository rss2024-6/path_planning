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


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
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

        # Dijkstra's Variables
        self.pos = [0, 0]
        self.map_data = np.array([0,0])

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
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data).reshape((h,w))
        blurred_data = ski.dilation(data, ski.square(15))
        cv2.imwrite('/home/racecar/racecar_ws/res_15.png',blurred_data)
        self.get_logger().info('Map Found!')
        self.map_data = list(blurred_data.flatten().astype('int8'))

    def pose_cb(self, pose):
        self.pos = [pose.pose.pose.position.x, pose.pose.pose.position.y]

    def goal_cb(self, msg):
        self.trajectory.clear()
        self.plan_path(self.pos, [msg.pose.position.x, msg.pose.position.y], self.map, self.map_data)

    def plan_path(self, start_point, end_point, map, mapdata):
        # extract info from map
        occupied = mapdata.copy()
        w = map.info.width
        h = map.info.height
        ori = map.info.origin.position
        res = map.info.resolution

        # choose number of points to sample and set up parameters
        length = len(occupied)
        self.get_logger().info('Begin Path Planning Process')
        num_pts = length // 500
        coords = [(x, y) for y in range(h) for x in range(w)]
        granularity = 20
        nn_radius = res*np.sqrt(w*h/num_pts)

        # removes start and end from sample options
        start_point = (float(start_point[0]), float(start_point[1]))
        end_point = (float(end_point[0]), float(end_point[1]))

        s = np.array(self.map_to_pixel(start_point))
        t = np.array(self.map_to_pixel(end_point))

        self.get_logger().info(str((s,t)))
        s_ind = s[1]*w+s[0]
        t_ind = t[1]*w+t[0]

        del coords[s_ind]
        del coords[t_ind]
        del occupied[s_ind]
        del occupied[t_ind]
        # yippee! wahoo! yay!

        coords = np.array(coords)
        occupied = np.array(occupied)
        # sample from likely unoccupied points
        likely = coords[occupied == 0]
        if len(likely) < num_pts:
            num_pts = len(likely)
        sample_inds = np.random.choice(np.arange(len(likely)), num_pts, replace = False)
        samples = likely[sample_inds]
        # self.get_logger().info(str(samples))

        # add grid cell coords of initial and goal pose to graph points
        pts = np.vstack((s, samples, t))
        num_pts = len(pts)


        # populate adjacency dictionary
        self.get_logger().info('wahoo! yippee! yay! pt1')
        adj = {i: {} for i in range(num_pts)}
        d_mtx = scipy.spatial.distance_matrix(pts, pts)*res
        edges = np.transpose((d_mtx < nn_radius).nonzero())
        for i, j in edges:
            if self.check_line(pts[i], pts[j], granularity):
                adj[i][j] = d_mtx[i][j]
        # self.get_logger().info(str(adj[0]))

        # convert from grid cell to world frame coords
        for i in range(num_pts):
            pts[i] = self.pixel_to_map(pts[i])
        pts[0]  = start_point
        pts[-1] = end_point

        # djikstra's
        fixed = {}
        node_dists = {i: np.inf for i in range(num_pts)}
        node_dists[0] = 0
        prev = {}
        prev[0] = 0
        
        reached = False
        self.get_logger().info('wahoo! yippee! yay! pt2')
        while not reached:
            ind = min(node_dists, key=node_dists.get)
            fixed[ind] = node_dists[ind]
            del node_dists[ind]
            for neighbor in adj[ind]:
                if neighbor in node_dists:
                    if (fixed[ind] + adj[ind][neighbor] < node_dists[neighbor]):
                        node_dists[neighbor] = fixed[ind] + adj[ind][neighbor]
                        prev[neighbor] = ind
            if ind == num_pts - 1:
                reached = True
        # self.get_logger().info(str(nn_radius))
        # self.get_logger().info(str(prev))
        
        # reconstruct path
        curr = num_pts - 1
        path = [curr]
        while curr != 0:
            curr = prev[curr]
            path.append(curr)
        path.reverse()
        self.get_logger().info(str(path))

        # check to shorten path
        #path_orig_len = len(path.copy())
        #for k in range(path_orig_len):
        #    for i in range(len(path)-2):
        #        if self.check_line(pts[path[i]], pts[path[i+2]], granularity*k):
        #            self.get_logger().info('removed %s' % path[i])
        #            path[i+1] = float('inf')
        #    path = list(filter(lambda pt: pt != float('inf'), path))

        # populate trajectory object
        for point in path:
            self.trajectory.addPoint(tuple(pts[point].astype(float)))
        

        # publish visual
        self.get_logger().info('wahoo! yippee! yay! pt3')
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()



    def check_line(self, pt1, pt2, granularity):
        w = self.map.info.width
        x = np.linspace(pt1[0], pt2[0], num = granularity, endpoint = True).astype(int)
        y = np.linspace(pt1[1], pt2[1], num = granularity, endpoint = True).astype(int)
        if self.map_data[pt1[1]*w+pt1[0]] != 0 or self.map_data[pt1[1]*w+pt1[0]] != 0:
            return False
        for i in range(len(x)):
            if self.map_data[y[i]*w+x[i]] != 0:
                return False
        return True

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

        return (int(scaled_x), int(scaled_y))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
