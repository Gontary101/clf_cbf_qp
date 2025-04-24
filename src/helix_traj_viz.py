#!/usr/bin/env python
# -*- coding: utf-8
import rospy
import ast
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose

class HelixTrajViz(object):
    def __init__(self):
        rospy.init_node('helix_traj_viz')

        # Publishers
        self.actual_pub  = rospy.Publisher('actual_traj_marker', Marker, queue_size=1)
        self.desired_pub = rospy.Publisher('desired_traj_marker', Marker, queue_size=1)
        self.obs_pub     = rospy.Publisher('obstacle_markers', MarkerArray, queue_size=1, latch=True)

        # Subscribers
        odom_topic = rospy.get_param('~odom_topic', '/iris/ground_truth/odometry')
        des_topic  = rospy.get_param('~des_topic',
                                     '/clf_iris_trajectory_controller/control/desired_position')
        rospy.Subscriber(odom_topic,  Odometry, self.odom_cb)
        rospy.Subscriber(des_topic,   Point,    self.desired_cb)

        # Prepare the “actual” trajectory Marker (green LINE_STRIP)
        self.actual = Marker(
            header=Header(frame_id='world'),
            ns='actual_traj',
            id=0,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            scale=Vector3(0.05, 0.0, 0.0),
            color=ColorRGBA(0.0, 1.0, 0.0, 1.0),
            points=[]
        )

        # Prepare the “desired” trajectory Marker (red LINE_STRIP)
        self.desired = Marker(
            header=Header(frame_id='world'),
            ns='desired_traj',
            id=1,
            type=Marker.LINE_STRIP,
            action=Marker.ADD,
            scale=Vector3(0.05, 0.0, 0.0),
            color=ColorRGBA(1.0, 0.0, 0.0, 1.0),
            points=[]
        )

        # Publish static obstacles once
        self.publish_obstacles()

        rospy.loginfo("HelixTrajViz ready – publishing markers.")
        rospy.spin()

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.actual.header.stamp = rospy.Time.now()
        self.actual.points.append(p)
        self.actual_pub.publish(self.actual)

    def desired_cb(self, msg):
        # msg is geometry_msgs/Point
        self.desired.header.stamp = rospy.Time.now()
        self.desired.points.append(msg)
        self.desired_pub.publish(self.desired)

    def publish_obstacles(self):
        obs_param = rospy.get_param('~static_obstacles',
                                    "[[-8.96, -15.52, 8.00, 1.00],"
                                    " [-7.92, 13.71, 13.00, 1.00],"
                                    " [13.75, 0.00, 18.00, 1.00],"
                                    " [-5.83, -10.10, 23.00, 1.00],"
                                    " [-4.79, 8.30, 28.00, 1.00]]")
        obs_list = ast.literal_eval(obs_param)
        ma = MarkerArray()
        for i, (x,y,z,r) in enumerate(obs_list):
            m = Marker(
                header=Header(frame_id='world'),
                ns='obstacles',
                id=i,
                type=Marker.SPHERE,
                action=Marker.ADD,
                pose=Pose(),  # placeholder
                scale=Vector3(r*2, r*2, r*2),
                color=ColorRGBA(1.0, 0.0, 0.0, 0.3)
            )
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            ma.markers.append(m)
        self.obs_pub.publish(ma)

if __name__ == '__main__':
    try:
        HelixTrajViz()
    except rospy.ROSInterruptException:
        pass
