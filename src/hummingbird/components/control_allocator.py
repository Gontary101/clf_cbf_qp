from clf_backstepping import CLFBackstepping

class ControlAllocator(object):
    def __init__(self, drone_model):
        self.clf = CLFBackstepping(drone_model)

    def compute_nominal_control(self, current_state_dict, reference_dict, gains):
        st = {
            "p_vec": current_state_dict["p_vec"],
            "v_vec": current_state_dict["v_world"],
            "phi": current_state_dict["phi"],
            "th": current_state_dict["th"],
            "psi": current_state_dict["psi"],
            "omega_body": current_state_dict["omega_b"],
            "R_mat": current_state_dict["R_mat"]
        }
        
        ref = {
            "tgt": reference_dict["tgt"],
            "vd": reference_dict["vd"],
            "ad": reference_dict["ad"],
            "yd": reference_dict["yd"],
            "rd": reference_dict["rd"],
            "t_traj_secs": reference_dict["t_traj_secs"],
            "state_name": reference_dict["state_name"]
        }
        
        return self.clf.compute(st, ref, gains)