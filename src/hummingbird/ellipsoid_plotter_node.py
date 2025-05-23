#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS node to visualize static box obstacles and their superellipsoid approximations.
Plots the box wireframes with low opacity and overlays the approximated ellipsoids
using matplotlib's 3D interactive plotting.
"""
import rospy
from gazebo_msgs.msg import ModelStates
from utils.obstacle_parser import GazeboObstacleProcessor
import numpy as np
import matplotlib
# Use an interactive backend suitable for your system
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import euler_from_quaternion


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    Taken from matplotlib cookbook.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def plot_box(ax, center, quat, size, color='gray', alpha=0.2):
    q = [quat.x, quat.y, quat.z, quat.w]
    _, _, yaw = euler_from_quaternion(q)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,            0,           1]])
    dx, dy, dz = size
    corners = np.array([[ dx/2,  dy/2,  dz/2],
                        [ dx/2,  dy/2, -dz/2],
                        [ dx/2, -dy/2,  dz/2],
                        [ dx/2, -dy/2, -dz/2],
                        [-dx/2,  dy/2,  dz/2],
                        [-dx/2,  dy/2, -dz/2],
                        [-dx/2, -dy/2,  dz/2],
                        [-dx/2, -dy/2, -dz/2]])
    wc = (R.dot(corners.T)).T + np.array([center.x, center.y, center.z])
    edges = [(0,1),(0,2),(0,4),(7,5),(7,6),(7,3),(1,3),(1,5),(2,3),(2,6),(4,5),(4,6)]
    for i,j in edges:
        ax.plot([wc[i,0], wc[j,0]],
                [wc[i,1], wc[j,1]],
                [wc[i,2], wc[j,2]],
                color=color, alpha=alpha)


class EllipsoidPlotter(object):
    def __init__(self):
        # Initialize ROS
        rospy.init_node('ellipsoid_plotter', anonymous=True)

        # precompute meshgrid for super‐ellipsoids once
        self.ell_u, self.ell_v = np.meshgrid(
            np.linspace(-np.pi/2, np.pi/2, rospy.get_param("~ellipsoid_resolution", 40)),
            np.linspace(-np.pi,   np.pi,   rospy.get_param("~ellipsoid_resolution", 60)),
        )

        # Read namespace & params for obstacle processor
        ns = rospy.get_param('~namespace', 'hummingbird')
        all_ns = rospy.get_param('~all_drone_namespaces', [])
        # load gazebo obstacle specs from global server (assumes you've rosparam-loaded obstacles.yaml)
        pget = lambda k, d: rospy.get_param(k, d)
        self.processor = GazeboObstacleProcessor(ns, all_ns, pget)

        # Prepare Matplotlib figure (main thread)
        plt.ion()
        self.fig = plt.figure('Obstacles vs Superellipsoids')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Shared data slot for latest ModelStates
        self.latest_msg = None

        # Subscribe in ROS, store messages
        topic = rospy.get_param('~model_states_topic', '/gazebo/model_states')
        rospy.Subscriber(topic, ModelStates, self.cb_models, queue_size=1)

        # Enter main update loop in *this* (main) thread
        self.run()

    def plot_superellipsoid(self, ax, center, a, b, c, n, R,
                            color='r', alpha=0.4):
        # Use precomputed meshgrids from the class instance
        u, v = self.ell_u, self.ell_v
        cu = np.sign(np.cos(u)) * (np.abs(np.cos(u)) ** (2.0 / n))
        su = np.sign(np.sin(u)) * (np.abs(np.sin(u)) ** (2.0 / n))
        cv = np.sign(np.cos(v)) * (np.abs(np.cos(v)) ** (2.0 / n))
        sv = np.sign(np.sin(v)) * (np.abs(np.sin(v)) ** (2.0 / n))
        # build in the *local* frame
        x_l = a * cu * cv
        y_l = b * cu * sv
        z_l = c * su
        # rotate to world and translate
        pts  = np.vstack((x_l.flatten(), y_l.flatten(), z_l.flatten()))
        pts  = R.dot(pts)
        x, y, z = (pts[0].reshape(x_l.shape) + center[0],
                   pts[1].reshape(y_l.shape) + center[1],
                   pts[2].reshape(z_l.shape) + center[2])
        ax.plot_surface(x, y, z, rstride=2, cstride=2,
                        color=color, alpha=alpha, linewidth=0)

    def cb_models(self, msg):
        # Called in ROS thread; just stash the latest message
        self.latest_msg = msg

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz update for plotting
        while not rospy.is_shutdown():
            msg = self.latest_msg
            if msg is not None:
                # process & redraw
                self.ax.cla()
                self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')

                # process obstacle models
                self.processor.process_model_states_msg(msg)

                for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
                    spec = self.processor.gz_shape_specs.get(name)
                    if not spec or spec.get('shape') != 'box':
                        continue
                    size = spec['size']
                    plot_box(self.ax, pose.position, pose.orientation, size)
                    ell = self.processor._ellipsoid_for_model(name, pose, twist)
                    cx, cy, cz  = ell[0:3]
                    a, b, c, n  = ell[9:13]
                    R_mat       = np.array(ell[13:22]).reshape(3, 3)
                    self.plot_superellipsoid(self.ax, (cx, cy, cz),
                                              a, b, c, n, R_mat)

                # equalize scale and draw
                set_axes_equal(self.ax)
                plt.draw()
            # pause to let GUI events process
            plt.pause(0.001)
            rate.sleep()


if __name__ == '__main__':
    try:
        EllipsoidPlotter()
    except rospy.ROSInterruptException:
        pass
