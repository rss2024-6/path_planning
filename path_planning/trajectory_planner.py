import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import tf_transformations as tfm
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

        # self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        # self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.odom_topic = "/odom"
        self.map_topic = "/map"
        self.initial_pose_topic = "/initialpose"
        self.map = OccupancyGrid()
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
        #blur = a = np.ones([10, 10], dtype = int) 
        #blurred_data = scipy.signal.convolve2d(data,blur)
        blurred_data = ski.dilation(data, ski.square(10))
        cv2.imwrite('/home/racecar/racecar_ws/res_10.png',blurred_data)
        self.get_logger().info('Map Found!')
        self.map_data = list(blurred_data.flatten().astype('int8'))

    def pose_cb(self, pose):
        self.pos = [pose.pose.pose.position.x, pose.pose.pose.position.y, 1]

    def goal_cb(self, msg):
        self.plan_path(self.pos, [msg.pose.position.x, msg.pose.position.y, 1], self.map, self.map_data)

    def check_line(self, pt1, pt2, thresh, granularity):
        w = self.map.info.width
        x = np.linspace(pt1[0], pt2[0], num = granularity, endpoint = True).astype(int)
        y = np.linspace(pt1[1], pt2[1], num = granularity, endpoint = True).astype(int)
        for i in range(len(x)):
            if self.map.data[y[i]*w+x[i]] > thresh:
                return False
        return True

    def plan_path(self, start_point, end_point, map, mapdata):
        # extract info from map
        occupied = mapdata
        w = map.info.width
        h = map.info.height
        ori = map.info.origin.position
        res = map.info.resolution
        ori_th = map.info.origin.orientation

        # choose number of points to sample and set up parameters
        length = len(occupied)
        self.get_logger().info('Begin Path Planning Process')
        num_pts = length // 2000
        coords = [(x, y, 1) for y in range(h) for x in range(w)]
        thresh = 0.25
        granularity = 200
        nn_radius = np.sqrt(w*h/num_pts)

        # removes start and end from sample options  ##TODO: ADD MAP ORIENTATION (see piazza post for transformation)
        T = tfm.quaternion_matrix([ori_th.x, ori_th.y, ori_th.z, ori_th.w])
        T = T[:3,:3]
        T[:,2] = np.array([ori.x,ori.y,1]).T
        s = (np.linalg.inv(T) @ np.array(start_point).T).T.astype(int)
        t = (np.linalg.inv(T) @ np.array(end_point  ).T).T.astype(int)
        #s = [int(- start_point[0]/res - ori.x), int(start_point[1]/res + ori.y)]
        #t = [int(- end_point[0]/res - ori.x), int(end_point[1]/res + ori.y)]
        self.get_logger().info(str((s,t)))
        s_ind = s[1]*w+s[0]
        t_ind = t[1]*w+t[0]
        del coords[s_ind]
        del coords[t_ind]
        del occupied[s_ind]
        del occupied[t_ind]
        # coords.remove(tuple(s))
        # coords.remove(tuple(t))
        coords = np.array(coords)
        # yippee! wahoo! yay!

        occupied = np.array(occupied)
        # sample from likely unoccupied points
        likely = coords[np.logical_and(occupied > -1, occupied < thresh)]
        if len(likely) < num_pts:
            num_pts = len(likely)
        sample_inds = np.random.choice(np.arange(len(likely)), num_pts, replace = False)
        samples = likely[sample_inds]

        # add grid cell coords of initial and goal pose to graph points
        pts = np.vstack((np.array([s]), samples, np.array([t])))
        # GRRRR
        num_pts = len(pts)

        # populate adjacency dictionary
        # adj = {i: {} for i in range(num_pts)}
        self.get_logger().info('wahoo! yippee! yay! pt1')
        # for i in range(num_pts):
        #     for j in range(i + 1, num_pts):
        #         if self.check_line(pts[i], pts[j], thresh, granularity):
        #             d = np.linalg.norm(pts[i] - pts[j])
        #             if d < nn_radius:
        #                 adj[i][j] = d * res
        #                 adj[j][i] = d * res
        # take 2
        adj = {i: {} for i in range(num_pts)}
        d_mtx = scipy.spatial.distance_matrix(pts, pts)*res
        edges = np.transpose((d_mtx < nn_radius).nonzero())
        for i, j in edges:
            if self.check_line(pts[i], pts[j], thresh, granularity):
                adj[i][j] = d_mtx[i][j]



        

        # convert from grid cell to world frame coords
        pts = (T @ pts.T).T
        pts[0]  = start_point
        pts[-1] = end_point
        #pts[:, 0] = -(pts[:, 0] + ori.x) * res
        #pts[:, 1] = (pts[:, 1] - ori.y) * res
        #pts[0] = start_point
        #pts[-1] = end_point

        # djikstra's
        fixed = {}
        node_dists = {i: np.inf for i in range(num_pts)}
        node_dists[0] = 0
        prev = {i: None for i in range(num_pts)}
        prev[0] = 0
        
        reached = False
        self.get_logger().info('wahoo! yippee! yay! pt2')
        while not reached:
            ind = min(node_dists, key=node_dists.get)
            fixed[ind] = node_dists[ind]
            del node_dists[ind]
            for neighbor in adj[ind]:
                if neighbor in node_dists:
                    # self.get_logger().info(str((node_dists[neighbor], fixed[ind], adj[ind][neighbor])))
                    # self.get_logger().info(str(adj[ind]))
                    node_dists[neighbor] = min(node_dists[neighbor], fixed[ind] + adj[ind][neighbor])
                    prev[neighbor] = ind
            if ind == num_pts - 1:
                reached = True

        # reconstruct path
        curr = num_pts - 1
        path = [curr]
        while curr != 0:
            curr = prev[curr]
            path.append(curr)
        path.reverse()

        # populate trajectory object
        for point in path:
            self.trajectory.addPoint(tuple(pts[point]))
        
        # publish visual
        self.get_logger().info('wahoo! yippee! yay! pt3')
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
