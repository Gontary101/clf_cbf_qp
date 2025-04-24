#!/usr/bin/env python
# -*- coding: utf-8
import rospy
import tf
from nav_msgs.msg import Odometry

if __name__ == '__main__':
    rospy.init_node('odom_to_tf')
    odom_topic = rospy.get_param('~odom_topic',
                                 '/iris/ground_truth/odometry')
    br = tf.TransformBroadcaster()

    def odom_cb(msg):
        # msg.header.frame_id = "world"
        # msg.child_frame_id  = "iris/base_link"
        br.sendTransform(
            (msg.pose.pose.position.x,
             msg.pose.pose.position.y,
             msg.pose.pose.position.z),
            (msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w),
            msg.header.stamp,
            msg.child_frame_id,
            msg.header.frame_id
        )

    rospy.Subscriber(odom_topic, Odometry, odom_cb, queue_size=1)
    rospy.spin()
