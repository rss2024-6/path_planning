import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, Pose, Point
from rclpy.node import Node
from tf_transformations import euler_from_quaternion, quaternion_from_euler


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

        self.last_found_index = 0

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #
        self.kp = 0 # for turning the bot

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.goal_pub = self.creat_publisher(Point, "/trajectory/goal_point", 1)

    def pose_callback(self, odometry_msg):
        pose_msg = odometry_msg.pose.pose
        position = pose_msg.position # robot current position
        quaterion = pose_msg.orientation
        theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2] # robot current heading

        # calculate center of the rear axle for center for the circle
        center = new Point()
        center.x = position.x - (self.wheelbase_length/2) * math.cos(math.radians(theta))
        center.y = position.y - (self.wheelbase_length/2) * math.sin(math.radians(theta))
        
        #last_found_index = self.last_found_index
        starting_index = self.last_found_index
        found_intersection = False
        goalPt = (0,0)
        path = self.trajectory.points

        # use loop to find intersections
        for i in range(starting_index, len(path)-1):
            # find intersection for the line between next 2 points on the path with the look-ahead circle
            x1 = path[i][0] - center.x
            y1 = path[i][1] - center.y
            x2 = path[i+1][0] - center.x
            y2 = path[i+1][1] - center.y
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt (dx**2 + dy**2)
            D = x1*y2 - x2*y1
            discriminant = (lookahead**2) * (dr**2) - D**2

            if discriminant >= 0: # if an intersection exists
                # find the intersections (solutions)
                sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                # intersections betwene circle and current path segment
                sol_pt1 = [sol_x1 + center.x, sol_y1 + center.y]
                sol_pt2 = [sol_x2 + center.y, sol_y2 + center.y]

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
                        if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
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
                    if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([center.x, center.y], path[i+1]):
                        # update lastFoundIndex and exit
                        self.lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        self.lastFoundIndex = i+1
                        # TODO log something here?

            else: # no intersections/solutions in range
                found_intersection = False
                # TODO: Log something here probably
                goalPt = [path[self.lastFoundIndex][0], path[self.lastFoundIndex][1]] # just try to go back to the last point idk??? TODO what to do in this case

        # publish the goal point
        goal_point = new Point()
        goal_point.x = goalPt[0]
        goal_point.y = goalPt[1]
        self.goal_pub.publish(goal_point)
        
        # calculate target angle from goal point
        target_angle = math.atan2(goalPt[1]-center.y, goalPt[0]-center.x) *180/math.pi
        if target_angle < 0: target_angle += 360

        steering_wheel_angle = math.atan(2*self.wheelbase*math.sin(target_angle)/self.lookahead)

        # calculate error from target angle
        #angle_error = target_angle - theta 
        #if angle_error >= 360 or angle_error <= -360:
        #    angle_error = angle_error % 360
        #if angle_error > 180:
        #    angle_error = angle_error - 360
        #elif angle_error < -180:
        #    angle_error = angle_error + 360

        # calculate desired turn velocity based on the angle error
        #turn_velocity = angle_error * self.kp

        # set and publish drive command
        drive = new AckermannDriveStamped()
        # TODO: drive.header? 
        drive.drive.speed = self.speed
        drive.drive.acceleration = 0
        drive.drive.jerk = 0
        drive.drive.steering_angle = steering_wheel_angle # TODO: what is the steering angle...
        #drive.drive.steering_angle_velocity = TODO
        self.drive_pub.publish()
        
    # params: intersection is tuple (x,y)
    #def publish_intersection(intersection):
        
    
    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

    # returns distance btwn 2 points
    def pt_to_pt_distance (pt1,pt2):
        distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        return distance

    # returns -1 if num is negative, 1 otherwise
    def sgn (num):
      if num >= 0:
        return 1
      else:
        return -1


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
