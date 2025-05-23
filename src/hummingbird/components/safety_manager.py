import rospy

# pick which filter backend via param: "zcbf", "c3bf", or "c3bf_adapter"
filter_type = rospy.get_param("safety_filter_type", "zcbf_ellipsoid").lower()

if filter_type == "c3bf_adapter":
    from components.c3bf_adapter import C3BFFilterAdapter as SAFETYFilter
elif filter_type == "c3bf":
    from obstacle_avoidance.C3BF_constraint import C3BFFilter as SAFETYFilter
elif filter_type == "zcbf_ellipsoid":
    from obstacle_avoidance.zcbf_constraint_ellipsoid import ZCBFFilter as SAFETYFilter
else:
    from obstacle_avoidance.zcbf_constraint import ZCBFFilter as SAFETYFilter
rospy.loginfo("Using %s filter", filter_type)
from cvxopt import matrix, solvers
import numpy as np
from std_msgs.msg import Float64MultiArray

class SafetyManager(object):
    def __init__(self, drone_model, obstacles_array, cbf_params, cbf_publisher):
        self.zcbf = SAFETYFilter(drone_model, obstacles_array, cbf_params,
                                 cbf_pub=cbf_publisher)

    def filter_control(self, state_name_str, U_nominal, current_kinematic_state,
                       clf_gains, clf_reference_dict):

        st_for_filter = current_kinematic_state.copy()

        if "v_world" in st_for_filter:
            st_for_filter["v_vec"] = st_for_filter.pop("v_world")

        if "omega_b" in st_for_filter:
            st_for_filter["omega_body"] = st_for_filter.pop("omega_b")

        st_for_filter['gains'] = clf_gains
        st_for_filter['ref'] = clf_reference_dict
        st_for_filter['ad_nom'] = clf_reference_dict["ad"]

        if state_name_str != "TRAJ":
            return U_nominal, None

        G_all, h_all = self.zcbf.constraints(st_for_filter)
        U_filtered = U_nominal
        slack_values = None

        if G_all.size > 0:
            P = matrix(np.eye(4))
            q_mat = matrix(-U_nominal)
            G_mat = matrix(G_all)
            h_mat = matrix(h_all)

            try:
                solvers.options['show_progress'] = rospy.get_param("~qp_solver_show_progress", False)
                sol = solvers.qp(P, q_mat, G_mat, h_mat)
                if sol['status'] == 'optimal':
                    U_filtered = np.asarray(sol['x']).flatten()
                    if self.zcbf.pub is not None:
                        slack_copt = h_mat - G_mat * sol['x']
                        slack_values = np.asarray(slack_copt).flatten()
                else:
                    rospy.logwarn_throttle(1.0,
                        "CLF-CBF QP returned %s - using nominal U", sol['status'])
            except ValueError:
                rospy.logwarn_throttle(1.0, "CLF-CBF QP infeasible - using nominal U")

        if self.zcbf.pub is not None:
            data_to_publish = [] if slack_values is None else list(slack_values)
            self.zcbf.pub.publish(Float64MultiArray(data=data_to_publish))

        return U_filtered, slack_values