import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
import math
import numpy as np

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

        self.last_found_index = 0

        self.lookahead = 0.7 
        self.speed = 1.  
        self.wheelbase_length = 0.5  
        self.turn_velocity = 0.4
        self.final_position = (0,0)
        self.stop = True

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback,1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.goal_pub = self.create_publisher(Marker, "/goal_point", 1)
        self.circle_pub = self.create_publisher(Marker, "/circle", 1)
        self.target_angle_pub = self.create_publisher(PoseArray, "/target_pose", 1)

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj or self.stop:
            drive = AckermannDriveStamped()
            drive.header.stamp = self.get_clock().now().to_msg()
            drive.header.frame_id = 'map'
            drive.drive.speed = 0.0
            drive.drive.acceleration = 0.0
            drive.drive.jerk = 0.0
            drive.drive.steering_angle = 0.0
            drive.drive.steering_angle_velocity = 0.0
            self.drive_pub.publish(drive)
            return

        pose_msg = odometry_msg.pose.pose
        position = pose_msg.position # robot current position
        quaternion = pose_msg.orientation
        theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2] # robot current heading
        path = self.trajectory.points

        # check if final_position is near robot, and stop robot if so
        circle_equation = (position.x - self.final_position[0])**2 + (position.y - self.final_position[1])**2 

        if circle_equation < 0.5 ** 2:
            self.stop = True
            drive = AckermannDriveStamped()
            drive.header.stamp = self.get_clock().now().to_msg()
            drive.header.frame_id = 'map'
            drive.drive.speed = 0.0
            drive.drive.acceleration = 0.0
            drive.drive.jerk = 0.0
            drive.drive.steering_angle = 0.0
            drive.drive.steering_angle_velocity = 0.0
            self.drive_pub.publish(drive)
            return

        self.publish_circle(position.x, position.y)
        
        starting_index = self.last_found_index
        found_intersection = False
        goalPt = (0,0)

        # use loop to find intersections
        for i in range(starting_index, len(path)-1):
            # find intersection for the line between next 2 points on the path with the look-ahead circle
            x1 = path[i][0] - position.x
            y1 = path[i][1] - position.y
            x2 = path[i+1][0] - position.x
            y2 = path[i+1][1] - position.y
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt (dx**2 + dy**2)
            D = x1*y2 - x2*y1
            discriminant = (self.lookahead**2) * (dr**2) - D**2

            if discriminant >= 0: # if an intersection exists
                # find the intersections (solutions)
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                # intersections betwene circle and current path segment
                sol_pt1 = [sol_x1 + position.x, sol_y1 + position.y]
                sol_pt2 = [sol_x2 + position.y, sol_y2 + position.y]

                # find the best/correct solution if there's 2
                minX = min(path[i][0], path[i+1][0])
                minY = min(path[i][1], path[i+1][1])
                maxX = max(path[i][0], path[i+1][0])
                maxY = max(path[i][1], path[i+1][1]) # bounding box for possible solutions

                # if either or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                    found_intersection = True

                    # if both solutions are in range, check which one is better
                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                        # make the decision by compare the distance between the intersections and the next point in path
                        if self.pt_to_pt_distance(sol_pt1, path[i+1]) < self.pt_to_pt_distance(sol_pt2, path[i+1]):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2
        
                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                        else:
                           goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if self.pt_to_pt_distance (goalPt, path[i+1]) < self.pt_to_pt_distance ([position.x, position.y], path[i+1]):
                        # update last_found_index and exit
                        self.last_found_index = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        self.last_found_index = i+1
                        self.get_logger().info("Intersections out of bounds. Perhaps drifted off the path?")

            else: # no intersections/solutions in range
                found_intersection = False
                goalPt = [path[self.last_found_index][0], path[self.last_found_index][1]] # just try to go back to the last point idk??? TODO what to do in this case

        # publish the current goal point
        self.publish_point(goalPt)

        # calculate target angle from car position to the goal point
        target_angle = math.atan2(goalPt[1]-position.y, goalPt[0]-position.x) #*180/math.pi
        self.publish_pose(position.x, position.y, target_angle)

        # calculate error from target angle
        angle_error = target_angle - theta 
        if angle_error >= 2*math.pi or angle_error <= -2*math.pi:
            angle_error = angle_error % 2*math.pi
        if angle_error > math.pi:
            angle_error = angle_error - 2*math.pi
        elif angle_error < -math.pi:
            angle_error = angle_error + 2*math.pi

        steering_wheel_angle = math.atan(2*self.wheelbase_length*math.sin(angle_error)/self.lookahead) 

        # set and publish drive command
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = 'map'
        drive.drive.speed = self.speed
        drive.drive.acceleration = 0.0
        drive.drive.jerk = 0.0
        drive.drive.steering_angle = steering_wheel_angle
        drive.drive.steering_angle_velocity = self.turn_velocity
        self.drive_pub.publish(drive)
    
    def publish_pose(self, x, y, theta):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        poses = []
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        q = quaternion_from_euler(0, 0, theta)
        quaternion = Quaternion()
        quaternion.x = q[0]
        quaternion.y = q[1]
        quaternion.z = q[2]
        quaternion.w = q[3]
        pose.orientation = quaternion

        poses.append(pose)

        msg.poses = poses
        self.target_angle_pub.publish(msg)
        
    # params: point is tuple (x,y)
    def publish_point(self, point):
        marker_msg = Marker()
        marker_msg.header.frame_id = 'map'  # Set the frame ID
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.type = Marker.POINTS
        marker_msg.action = Marker.ADD
        marker_msg.scale.x = 0.2  # Set the scale of the points
        marker_msg.scale.y = 0.2
        marker_msg.scale.z = 0.2
        marker_msg.color.a = 1.0  # Set the alpha value (transparency)
        marker_msg.color.g = 1.0  # Set the color to green

        p = Point()
        p.x = point[0] * 1.0
        p.y = point[1] * 1.0
        marker_msg.points.append(p)
        self.goal_pub.publish(marker_msg)
    
    def publish_circle(self, x, y):
        marker_msg = Marker()
        marker_msg.header.frame_id = 'map'  # Set the frame ID
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.type = Marker.POINTS
        marker_msg.action = Marker.ADD
        marker_msg.scale.x = 0.05  # Set the scale of the points
        marker_msg.scale.y = 0.05
        marker_msg.scale.z = 0.05
        marker_msg.color.a = 1.0  # Set the alpha value (transparency)
        marker_msg.color.b = 1.0  # Set the color to blue
        
        for i in range(36):
            px = x+ self.lookahead * math.cos(i * math.pi/18)
            py = y+self.lookahead * math.sin(i * math.pi/18)
            point = Point()
            point.x = px
            point.y = py
            marker_msg.points.append(point)

        center = Point()
        center.x = x
        center.y = y
        marker_msg.points.append(center)

        self.circle_pub.publish(marker_msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.last_found_index=0
        self.final_position = self.trajectory.points[-1]

        self.initialized_traj = True
        self.stop = False

    # returns distance btwn 2 points
    def pt_to_pt_distance (self, pt1,pt2):
        distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        return distance

    # returns -1 if num is negative, 1 otherwise
    def sgn (self, num):
      if num >= 0:
        return 1
      else:
        return -1


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
