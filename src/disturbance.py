#!/usr/bin/env python
import rospy
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
import time
import random # Import the random module

# --- Function to Apply Wrench ---
def apply_disturbance(body_name, force, torque, duration_sec):
    """
    Applies a specified force and torque (wrench) to a body in Gazebo
    for a given duration using the /gazebo/apply_body_wrench service.

    Args:
        body_name (str): The name of the body link in Gazebo (e.g., "model_name::link_name").
        force (list[float]): A list of 3 floats representing the force [fx, fy, fz] in Newtons.
        torque (list[float]): A list of 3 floats representing the torque [tx, ty, tz] in Newton-meters.
        duration_sec (float): The duration for which the wrench should be applied, in seconds.
    """
    # Wait for the service to become available
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    try:
        # Create a service proxy
        apply_wrench_service = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # Create the request message
        req = ApplyBodyWrenchRequest()
        req.body_name = body_name
        # Apply wrench in the world frame for consistent direction
        req.reference_frame = "world"
        # Set the force components
        req.wrench.force.x = force[0]
        req.wrench.force.y = force[1]
        req.wrench.force.z = force[2]
        # Set the torque components
        req.wrench.torque.x = torque[0]
        req.wrench.torque.y = torque[1]
        req.wrench.torque.z = torque[2]
        # Apply the wrench immediately
        req.start_time = rospy.Time(0)
        # Set the duration for the wrench application
        req.duration = rospy.Duration(duration_sec)

        # Call the service
        resp = apply_wrench_service(req)

        # Log the outcome using .format() for compatibility
        if resp.success:
            log_msg = "Applied disturbance wrench to {} (Force Z: {:.2f} N) for {:.2f} sec".format(
                body_name, force[2], duration_sec
            )
            rospy.loginfo(log_msg)
        else:
            log_msg = "Failed to apply disturbance wrench to {}: {}".format(
                body_name, resp.status_message
            )
            rospy.logwarn(log_msg)

    except rospy.ServiceException as e:
        # Log service errors using .format()
        rospy.logerr("Service call failed: {}".format(e))
    except Exception as e:
        # Log other errors using .format()
        rospy.logerr("An unexpected error occurred in apply_disturbance: {}".format(e))

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        # Initialize the ROS node
        rospy.init_node('disturbance_injector_random_z')
        rospy.loginfo("Random Z-Disturbance Injector Node Started")

        # --- Configuration ---
        target_body = "iris/base_link" # IMPORTANT: Verify this matches your Gazebo model::link name exactly!
        min_force_z = -5.0           # Minimum Z-force in Newtons
        max_force_z = 5.0            # Maximum Z-force in Newtons
        disturbance_interval = 6.0  # Time between disturbances in seconds
        disturbance_duration = 1.5   # Duration of each disturbance pulse in seconds
        initial_delay = 10.0         # Wait for simulation to potentially stabilize

        # Log configuration using .format()
        rospy.loginfo("Target body: {}".format(target_body))
        rospy.loginfo("Applying random Z-force between {} N and {} N".format(min_force_z, max_force_z))
        rospy.loginfo("Disturbance interval: {} s, Duration: {} s".format(disturbance_interval, disturbance_duration))

        # Wait for Gazebo and the model to be ready
        rospy.loginfo("Waiting {} seconds before starting disturbances...".format(initial_delay))
        rospy.sleep(initial_delay)
        rospy.loginfo("Starting disturbance loop.")

        # --- Main Loop ---
        while not rospy.is_shutdown():
            # Generate random Z-force within the specified range
            random_force_z = random.uniform(min_force_z, max_force_z)

            # Define the force vector (only Z component is non-zero)
            force_to_apply = [0.0, 0.0, random_force_z]
            # Define the torque vector (no torque applied)
            torque_to_apply = [0.0, 0.0, 0.0]

            # Apply the generated disturbance
            apply_disturbance(target_body, force_to_apply, torque_to_apply, disturbance_duration)

            # Wait for the specified interval before the next disturbance
            rospy.sleep(disturbance_interval)

    except rospy.ROSInterruptException:
        rospy.loginfo("Disturbance Injector node interrupted and shut down.")
    except Exception as e:
        # Log critical errors using .format()
        rospy.logerr("An critical error occurred in the main loop: {}".format(e))
    finally:
        rospy.loginfo("Disturbance Injector Node Finished.")

