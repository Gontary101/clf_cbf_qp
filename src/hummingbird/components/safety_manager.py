import rospy

# pick which filter backend via param: "zcbf", "c3bf", or "c3bf_adapter", or "zcbf_ellipsoid"
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
        
        # Add debug logging initialization
        self._last_debug_log_time = rospy.get_time()
        self._debug_log_interval = 0.5  # Log every 0.5 seconds

    def filter_control(self, state_name_str, U_nominal, current_kinematic_state,
                       clf_gains, clf_reference_dict):
        
        current_time = rospy.get_time()
        should_log_debug = (current_time - self._last_debug_log_time) >= self._debug_log_interval

        st_for_filter = current_kinematic_state.copy()

        if "v_world" in st_for_filter:
            st_for_filter["v_vec"] = st_for_filter.pop("v_world")

        if "omega_b" in st_for_filter:
            st_for_filter["omega_body"] = st_for_filter.pop("omega_b")

        st_for_filter['gains'] = clf_gains
        st_for_filter['ref'] = clf_reference_dict
        st_for_filter['ad_nom'] = clf_reference_dict["ad"]

        if state_name_str != "TRAJ":
            if should_log_debug:
                rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] State '%s' != 'TRAJ', skipping safety filter", state_name_str)
                self._last_debug_log_time = current_time
            return U_nominal, None

        G_all, h_all = self.zcbf.constraints(st_for_filter)
        U_filtered = U_nominal
        slack_values = None
        
        # Debug logging for constraint matrices
        if should_log_debug:
            cbf_constraints = G_all.shape[0] - 2 if G_all.size > 0 else 0  # -2 for box constraints
            rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] Generated %d CBF constraints (+ 2 box constraints)", cbf_constraints)
            
            if G_all.size > 0:
                rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] Constraint matrix G shape: %s, h vector shape: %s", 
                                     str(G_all.shape), str(h_all.shape))
                
                # Log constraint values summary
                h_min, h_max = np.min(h_all), np.max(h_all)
                G_norm = np.linalg.norm(G_all, axis=1)  # norm of each row
                G_norm_min, G_norm_max = np.min(G_norm), np.max(G_norm)
                rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] h_range=[%.4f, %.4f], G_row_norms=[%.4f, %.4f]", 
                                     h_min, h_max, G_norm_min, G_norm_max)

        if G_all.size > 0:
            P = matrix(np.eye(4))
            q_mat = matrix(-U_nominal)
            G_mat = matrix(G_all)
            h_mat = matrix(h_all)

            try:
                solvers.options['show_progress'] = rospy.get_param("~qp_solver_show_progress", False)
                
                if should_log_debug:
                    rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] Solving QP with U_nominal=[%.3f,%.3f,%.3f,%.3f]", 
                                         U_nominal[0], U_nominal[1], U_nominal[2], U_nominal[3])
                
                sol = solvers.qp(P, q_mat, G_mat, h_mat)
                
                if sol['status'] == 'optimal':
                    U_filtered = np.asarray(sol['x']).flatten()
                    
                    if should_log_debug:
                        U_diff_norm = np.linalg.norm(U_filtered - U_nominal)
                        rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] QP solved optimally, U_change_norm=%.6f", U_diff_norm)
                        
                        # Log QP solution details
                        if 'primal objective' in sol:
                            rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] QP objective value: %.6f", sol['primal objective'])
                    
                    if self.zcbf.pub is not None:
                        slack_copt = h_mat - G_mat * sol['x']
                        slack_values = np.asarray(slack_copt).flatten()
                        
                        if should_log_debug:
                            slack_min, slack_max = np.min(slack_values), np.max(slack_values)
                            violated_constraints = np.sum(slack_values < 1e-6)
                            rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] Slack: min=%.6f, max=%.6f, violated_count=%d", 
                                                 slack_min, slack_max, violated_constraints)
                else:
                    rospy.logwarn_throttle(1.0,
                        "[SAFETY_MANAGER_DEBUG] CLF-CBF QP returned %s - using nominal U", sol['status'])
                    if should_log_debug:
                        rospy.logwarn_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] QP failed with status: %s", sol['status'])
                        
            except ValueError as e:
                rospy.logwarn_throttle(1.0, "[SAFETY_MANAGER_DEBUG] CLF-CBF QP infeasible - using nominal U. Error: %s", str(e))
                if should_log_debug:
                    rospy.logwarn_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] QP ValueError: %s", str(e))
                    
            except Exception as e:
                rospy.logerr_throttle(1.0, "[SAFETY_MANAGER_DEBUG] Unexpected QP error: %s", str(e))
                if should_log_debug:
                    rospy.logerr_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] Unexpected QP error: %s", str(e))

        else:
            if should_log_debug:
                rospy.loginfo_throttle(self._debug_log_interval, "[SAFETY_MANAGER_DEBUG] No constraints generated, using nominal control")

        if should_log_debug:
            self._last_debug_log_time = current_time

        if self.zcbf.pub is not None:
            data_to_publish = [] if slack_values is None else list(slack_values)
            self.zcbf.pub.publish(Float64MultiArray(data=data_to_publish))

        return U_filtered, slack_values