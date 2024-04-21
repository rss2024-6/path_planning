import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np


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
        self.pos = [0, 0]

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

    def pose_cb(self, pose):
        self.pos = [pose.position.x, pose.position.y]

    def goal_cb(self, msg):
        self.plan_path(self.pos, [msg.position.x, msg.position.y], self.map)

    def check_line(self, pt1, pt2, thresh, granularity):
        x = np.linspace(pt1[0], pt2[0], num = granularity, endpoint = True)
        y = np.linspace(pt1[1], pt2[1], num = granularity, endpoint = True)
        for i in range(len(x)):
            if self.map.data[y[i]][x[i]] > thresh:
                return False
        return True

    def plan_path(self, start_point, end_point, map):
        # extract info from map
        occupied = map.data
        w = map.info.width
        h = map.info.height
        ori = map.info.origin.position
        res = map.info.resolution

        # choose number of points to sample and set up parameters
        length = len(occupied)
        num_pts = length // 20
        coords = [[x, y] for y in range(h) for x in range(w)]
        thresh = 0.25
        granularity = 5
        nn_radius = 3*np.sqrt(w*h/num_pts)

        # removes start and end from sample options
        s = [int(- start_point[0]/res - ori.x), int(start_point[1]/res + ori.y)]
        t = [int(- end_point[0]/res - ori.x), int(end_point[1]/res + ori.y)]
        coords.remove(s)
        coords.remove(t)
        coords = np.array(coords)

        # sample from likely unoccupied points proportional to probability of being unoccupied
        likely = coords[occupied < thresh]
        if len(likely) < num_pts:
            num_pts = len(likely)
        samples = np.random.choice(likely, num_pts, p = occupied[likely], replace = False)

        # add grid cell coords of initial and goal pose to graph points
        pts = np.vstack(np.array([s]), samples, np.array([t]))
        num_pts = len(pts)

        # populate adjacency dictionary
        adj = {i: {} for i in range(num_pts)}
        for i in range(num_pts):
            for j in range(i + 1, num_pts):
                if self.check_line(pts[i], pts[j], thresh, granularity):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < nn_radius:
                        adj[i][j] = d * res
                        adj[j][i] = d * res

        # convert from grid cell to world frame coords
        pts[:, 0] = -(pts[:, 0] + ori.x) * res
        pts[:, 1] = (pts[:, 1] - ori.y) * res
        pts[0] = start_point
        pts[-1] = end_point

        # djikstra's
        fixed = {}
        node_dists = {i: np.inf for i in range(num_pts)}
        node_dists[0] = 0
        prev = {i: None for i in range(num_pts)}
        prev[0] = 0
        
        reached = False
        while not reached:
            ind = min(node_dists, key=node_dists.get)
            fixed[ind] = node_dists[ind]
            del node_dists[ind]
            for neighbor in adj[ind]:
                if neighbor in node_dists:
                    node_dists[neighbor] = min(node_dists[neighbor], fixed[ind] + adj[ind][neighbor])
                    prev[neighbor] = ind
            if ind == num_pts + 1:
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
            self.trajectory.addPoint(tuple(samples[point]))
        
        # publish visual
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
