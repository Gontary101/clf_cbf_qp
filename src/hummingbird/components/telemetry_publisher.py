import math
import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String

class TelemetryPublisher(object):
    def __init__(self):
        misc_topics = [
            ("control/state", String),
            ("control/U", Float64MultiArray),
            ("control/omega_sq", Float64MultiArray),
            ("error/position", Point),
            ("error/velocity", Vector3),
            ("error/attitude_deg", Point),
            ("error/rates_deg_s", Vector3),
            ("control/desired_position", Point),
            ("control/desired_velocity", Vector3),
            ("control/desired_acceleration", Vector3),
            ("control/desired_attitude_deg", Point),
            ("control/virtual_inputs", Point),
            ("state/position", Point),          
            ("state/velocity_world", Vector3),         
            ("state/orientation_eul", Point),           
            ("state/rates_body", Vector3),         
            ("control/U_nominal", Float64MultiArray)
        ]
        q_size = rospy.get_param("~telemetry_publisher_queue_size", 1)
        self.pubs = {n: rospy.Publisher("~" + n, m, queue_size=q_size)
                     for n, m in misc_topics}

    def publish_telemetry(self, current_kinematics, flight_state_name,
                          U_final, w_sq_final, clf_output_dict,
                          reference_signals, U_nominal_clf):

        self.pubs["state/position"].publish(Point(*current_kinematics["p_vec"]))
        self.pubs["state/velocity_world"].publish(Vector3(*current_kinematics["v_world"]))
        self.pubs["state/orientation_eul"].publish(Point(current_kinematics["phi"], current_kinematics["th"], current_kinematics["psi"]))
        self.pubs["state/rates_body"].publish(Vector3(*current_kinematics["omega_b"]))
        
        self.pubs["control/U_nominal"].publish(Float64MultiArray(data=U_nominal_clf))
        self.pubs["control/state"].publish(String(data=flight_state_name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U_final))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq_final))
        
        self.pubs["error/position"].publish(Point(*clf_output_dict["ex1"]))
        self.pubs["error/velocity"].publish(Vector3(*clf_output_dict["ex2"]))
        self.pubs["error/attitude_deg"].publish(
            Point(*(math.degrees(i) for i in clf_output_dict["e_th"])))
        self.pubs["error/rates_deg_s"].publish(
            Vector3(*(math.degrees(i) for i in clf_output_dict["e_w"])))
            
        self.pubs["control/desired_position"].publish(Point(*reference_signals["tgt"]))
        self.pubs["control/desired_velocity"].publish(Vector3(*reference_signals["vd"]))
        self.pubs["control/desired_acceleration"].publish(Vector3(*reference_signals["ad"]))
        self.pubs["control/desired_attitude_deg"].publish(
            Point(*(math.degrees(i) for i in
                    (clf_output_dict["phi_d"], clf_output_dict["theta_d"], reference_signals["yd"]))))
        self.pubs["control/virtual_inputs"].publish(
            Point(clf_output_dict["Uex"], clf_output_dict["Uey"], 0.0))