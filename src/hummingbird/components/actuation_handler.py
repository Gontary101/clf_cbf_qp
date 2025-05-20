from mav_msgs.msg import Actuators

class ActuationHandler(object):
    def __init__(self, drone_model, cmd_publisher):
        self.model = drone_model
        self.cmd_pub = cmd_publisher

    def generate_and_publish_motor_commands(self, U_control_input, timestamp):
        w_cmd, w_sq = self.model.thrust_torques_to_motor_speeds(U_control_input)
        
        m_msg = Actuators()
        m_msg.header.stamp = timestamp
        m_msg.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m_msg)
        
        return w_cmd, w_sq

    def send_single_stop_command(self):
        stop_msg = Actuators()
        stop_msg.angular_velocities = [0.0] * 4
        self.cmd_pub.publish(stop_msg)