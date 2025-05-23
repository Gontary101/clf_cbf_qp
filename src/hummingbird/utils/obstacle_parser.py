import ast
import numpy as np
import rospy
import math
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates
from .dynamics_utils2 import rotation_matrix


def parse_obstacles(pget_func):
    default_obs_str = "[]"
    try:
        raw = pget_func("dynamic_obstacles", default_obs_str)
        lst = ast.literal_eval(raw)
        if not isinstance(lst, list):
            rospy.logwarn("Parsed dynamic_obstacles is not a list. Using empty list.")
            lst = []
        
        if lst and not all(
                isinstance(o, (list, tuple)) and len(o) == 10 and
                all(isinstance(n, (int, float)) for n in o)
                for o in lst):
            rospy.logwarn("Invalid format in dynamic_obstacles. Expected list of 10-element lists/tuples. Using empty list.")
            lst = []

        obs_array = np.array(lst, dtype=float)
        if obs_array.ndim == 1 and obs_array.size == 0:
             obs_array = np.empty((0, 10), dtype=float)
        elif obs_array.ndim == 1 and obs_array.shape[0] == 10:
            obs_array = obs_array.reshape(1, 10)
        elif obs_array.ndim != 2 or (obs_array.size > 0 and obs_array.shape[1] != 10):
            rospy.logwarn("Parsed dynamic_obstacles does not have shape (N, 10). Using empty list.")
            obs_array = np.empty((0,10), dtype=float)
        
        if obs_array.size > 0:
            rospy.loginfo("Loaded %d dynamic obstacles.", obs_array.shape[0])
        else:
            rospy.loginfo("No dynamic obstacles loaded or an error occurred in parsing.")
        return obs_array

    except (ValueError, SyntaxError) as e:
         rospy.logwarn("Error parsing dynamic_obstacles parameter '%s': %s. Using empty list.", raw, e)
         return np.empty((0,10), dtype=float)
    except Exception as e:
         rospy.logwarn("Unexpected error processing dynamic_obstacles '%s': %s. Using empty list.", raw, e)
         return np.empty((0,10), dtype=float)


class GazeboObstacleProcessor(object):
    def __init__(self, ns, all_drone_namespaces, pget_func):
        self.ns = ns
        self.all_drone_namespaces = all_drone_namespaces
        self.pget_func = pget_func

        # Fetch and store parameters
        raw_gz_shape_specs = self.pget_func("gazebo_obstacle_shapes", {}) # Default to empty dict
        if isinstance(raw_gz_shape_specs, dict):
            self.gz_shape_specs = raw_gz_shape_specs
        elif isinstance(raw_gz_shape_specs, str):
            try:
                parsed_spec = ast.literal_eval(raw_gz_shape_specs)
                if isinstance(parsed_spec, dict):
                    self.gz_shape_specs = parsed_spec
                else:
                    rospy.logwarn("gazebo_obstacle_shapes (string) did not evaluate to a dict, using empty. Value: %s", raw_gz_shape_specs)
                    self.gz_shape_specs = {}
            except (ValueError, SyntaxError) as e:
                rospy.logwarn("Error parsing gazebo_obstacle_shapes (string): %s. Using empty dict. Value: %s", e, raw_gz_shape_specs)
                self.gz_shape_specs = {}
        else:
            rospy.logwarn("gazebo_obstacle_shapes is neither a dict nor a string, using empty. Type: %s", type(raw_gz_shape_specs))
            self.gz_shape_specs = {}

        # If the parameter didn't load as a dict, try to get nested parameters
        if not self.gz_shape_specs:
            try:
                # Try to get all parameters under gazebo_obstacle_shapes/
                all_params = rospy.get_param_names()
                gz_prefix = rospy.get_name() + "/gazebo_obstacle_shapes/"
                
                # Find all parameters that start with the gazebo_obstacle_shapes prefix
                gz_params = [p for p in all_params if p.startswith(gz_prefix)]
                
                # Build the nested dictionary structure
                gz_shapes = {}
                for param_name in gz_params:
                    # Remove the prefix to get the relative path
                    rel_path = param_name[len(gz_prefix):]
                    path_parts = rel_path.split('/')
                    
                    if len(path_parts) >= 2:  # Should be model_name/property
                        model_name = path_parts[0]
                        prop_name = path_parts[1]
                        
                        if model_name not in gz_shapes:
                            gz_shapes[model_name] = {}
                        
                        # Get the parameter value
                        param_value = rospy.get_param(param_name)
                        gz_shapes[model_name][prop_name] = param_value
                
                self.gz_shape_specs = gz_shapes
                
                if self.gz_shape_specs:
                    rospy.loginfo("[%s] Loaded %d gazebo obstacle shapes from nested parameters", ns, len(self.gz_shape_specs))
                
            except Exception as e:
                rospy.logwarn("Error loading nested gazebo_obstacle_shapes parameters: %s", e)
                self.gz_shape_specs = {}

        if not self.gz_shape_specs:
             rospy.loginfo_once("[%s] GazeboObstacleProcessor: No Gazebo shape specifications loaded or parsing failed, will use defaults for unknown models." % ns)

        self.max_spheres = int(self.pget_func("zcbf_max_spheres", 150))
        self.default_obs_r = float(self.pget_func("default_obstacle_radius", 0.35))
        self.decomp_r_default = float(self.pget_func("sphere_decomp_r", 0.25))
        self.obs_inflation = float(self.pget_func("obstacle_inflation", 0.30))
        # This drone_radius is for OTHER drones when they are obstacles
        self.drone_radius = float(self.pget_func("drone_radius", 0.3)) 

        self.gz_obstacle_spheres = np.empty((0, 10), dtype=float)
        # will hold our ellipsoid approximations: [x,y,z,vx,vy,vz,ax,ay,az,a,b,c,n]
        self.gz_ellip_obs       = np.empty((0,13), dtype=float)
        # figure out which mode we're in
        self._shape_approx      = pget_func("safety_filter_type", "zcbf").lower()
        
        self.other_drone_states = {
            other_ns: {'pose': None, 'twist': None, 'time': rospy.Time(0)}
            for other_ns in self.all_drone_namespaces if other_ns != self.ns
        }
        self.state_timeout = rospy.Duration(float(self.pget_func("state_timeout_secs", 1.0)))

    def _rot_mat_from_quat(self, q_ros):
        q_tf = [q_ros.x, q_ros.y, q_ros.z, q_ros.w]
        roll, pitch, yaw = euler_from_quaternion(q_tf)
        return rotation_matrix(roll, pitch, yaw)


    def _box_to_spheres(self, size, r_sphere):
        lx, ly, lz = size
        nx = max(1, int(round(lx / (2 * r_sphere))))
        ny = max(1, int(round(ly / (2 * r_sphere))))
        nz = max(1, int(round(lz / (2 * r_sphere))))

        # Spacing between sphere centers
        dx = lx / nx if nx > 0 else 0
        dy = ly / ny if ny > 0 else 0
        dz = lz / nz if nz > 0 else 0

        # Create sphere centers
        centers = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Center of each sphere in local frame of the box
                    local_x = -lx/2 + dx/2 + i*dx
                    local_y = -ly/2 + dy/2 + j*dy
                    local_z = -lz/2 + dz/2 + k*dz
                    centers.append([local_x, local_y, local_z])
        
        if not centers: # Should not happen if nx,ny,nz >=1
            return np.empty((0,3))
            
        return np.array(centers)

    def _spheres_for_model(self, model_name, model_pose_ros, r_sub_decomp):
        spheres = []
        pos_global = np.array([model_pose_ros.position.x,
                               model_pose_ros.position.y,
                               model_pose_ros.position.z])
        rot_global = self._rot_mat_from_quat(model_pose_ros.orientation)

        spec = self.gz_shape_specs.get(model_name, {})
        geom_type = spec.get('shape', 'sphere')

        if geom_type == 'box' and 'size' in spec:
            # Inflate the box size first
            size_inflated = np.array(spec['size'], dtype=float) + 2 * self.obs_inflation
            for lc in self._box_to_spheres(size_inflated, r_sub_decomp):
                gc = pos_global + rot_global.dot(lc)
                final_r = r_sub_decomp + self.obs_inflation
                spheres.append([gc[0], gc[1], gc[2], 0,0,0, 0,0,0, final_r])
        else:
            radius_as_primitive = float(spec.get('radius', self.default_obs_r)) 
            final_radius = radius_as_primitive + self.obs_inflation
            spheres.append([pos_global[0], pos_global[1], pos_global[2], 0,0,0, 0,0,0, final_radius])
            
        return spheres

    def _ellipsoid_for_model(self, model_name, pose_ros, twist_ros):
        spec = self.gz_shape_specs.get(model_name, {})
        shape = spec.get("shape", "sphere")
        # enlarge every semiaxis by (obs_inflation + r_drone) so that
        # the CBF works with the *true* keep-out set
        safety_buffer = self.obs_inflation + self.drone_radius

        if shape=="box" and "size" in spec:
            dx, dy, dz = spec["size"]
            a = dx/2 + safety_buffer
            b = dy/2 + safety_buffer
            c = dz/2 + safety_buffer
            # exponent n based on aspect ratio, clamped to a maximum
            rat   = max(dx, dy, dz) / max(min(dx, dy, dz), 1e-6)
            n_raw = max(2, int(round(rat)))
            # allow override via ROS param, default to 6
            max_n = int(self.pget_func("max_ellipsoid_exponent", 6))
            n     = min(n_raw, max_n)
        else:
            r = float(spec.get("radius", self.default_obs_r)) + safety_buffer
            a=b=c=r;  n=2

        x,y,z = (pose_ros.position.x,
                 pose_ros.position.y,
                 pose_ros.position.z)
        vx,vy,vz = (twist_ros.linear.x,
                    twist_ros.linear.y,
                    twist_ros.linear.z)

        rot = self._rot_mat_from_quat(pose_ros.orientation)
        rot_flat = rot.reshape(9).tolist()
        return [ x,y,z,
                 vx,vy,vz,
                 0.0,0.0,0.0,
                 a,b,c,n
               ] + rot_flat

    def process_model_states_msg(self, msg):
        tmp_spheres = []
        tmp_ellip   = []
        now = rospy.Time.now() # For updating other_drone_states timestamp

        for i, name in enumerate(msg.name):
            if name == self.ns: # Skip ego drone
                continue

            if name in self.other_drone_states:
                self.other_drone_states[name]['pose'] = msg.pose[i]
                self.other_drone_states[name]['twist'] = msg.twist[i]
                self.other_drone_states[name]['time'] = now
            else:
                model_spec = self.gz_shape_specs.get(name, {})
                geom_type = model_spec.get('type', 'sphere') # Default to sphere

                if geom_type == 'box':
                    r_for_decomp = float(model_spec.get('sphere_r', self.decomp_r_default))
                else: # sphere or other
                    r_for_decomp = float(model_spec.get('radius', self.default_obs_r))

                # old sphere-decomp
                model_spheres = self._spheres_for_model(name, msg.pose[i], r_for_decomp)
                tmp_spheres.extend(model_spheres)
                # new single-ellipsoid approximation
                tmp_ellip.append(
                    self._ellipsoid_for_model(name, msg.pose[i], msg.twist[i])
                )

        # store both representations
        self.gz_obstacle_spheres = np.array(tmp_spheres, dtype=float) \
            if tmp_spheres else np.empty((0,10),dtype=float)
        self.gz_ellip_obs       = np.array(tmp_ellip,   dtype=float) \
            if tmp_ellip   else np.empty((0,22),dtype=float)

    def get_combined_obstacles(self, current_drone_p_vec_numpy):
        now = rospy.Time.now()
        dynamic_obs_list = []
        dynamic_elip_list = []

        # Dynamic drone obstacles
        for other_ns, state_data in self.other_drone_states.items():
            if state_data['pose'] and state_data['twist'] and (now - state_data['time'] < self.state_timeout):
                op = state_data['pose'].position
                ov = state_data['twist'].linear
                # sphere for other-drone
                r = self.drone_radius + self.obs_inflation
                dynamic_obs_list.append([
                    op.x, op.y, op.z,
                    ov.x, ov.y, ov.z,
                    0,0,0,
                    r
                ])
                # ellipsoid for other-drone (a=b=c=r, n=2) + identity rotation matrix
                identity_rot_flat = np.eye(3).reshape(9).tolist()
                dynamic_elip_list.append([
                    op.x, op.y, op.z,
                    ov.x, ov.y, ov.z,
                    0,0,0,
                    r, r, r, 2
                ] + identity_rot_flat)
        
        dynamic_obs_array   = np.array(dynamic_obs_list, dtype=float) \
            if dynamic_obs_list else np.empty((0,10), dtype=float)
        dynamic_elip_array  = np.array(dynamic_elip_list, dtype=float) \
            if dynamic_elip_list else np.empty((0,22), dtype=float)

        if self._shape_approx == "zcbf_ellipsoid":
            parts = [p for p in (dynamic_elip_array, self.gz_ellip_obs) if p.size>0]
        else:
            parts = [p for p in (dynamic_obs_array, self.gz_obstacle_spheres) if p.size>0]

        if parts:
            combined_obs = np.vstack(parts)
        else:
            combined_obs = np.empty((0,10 if self._shape_approx!="zcbf_ellipsoid" else 22), dtype=float)

        if combined_obs.size == 0:
            return combined_obs

        # Filtering
        z_ground_tol = float(self.pget_func("z_ground_tol", 0.0)) # Ground clearance
        cbf_active_range = float(self.pget_func("cbf_active_range_ellipse", 2.0)) # Max range to consider obstacles
        max_active_spheres = int(self.pget_func("zcbf_max_active_spheres", 10)) # Max obstacles to feed to CBF
        combined_obs = combined_obs[combined_obs[:,2] > z_ground_tol]
        if combined_obs.size == 0:
            return combined_obs
        delta_pos = combined_obs[:, :3] - current_drone_p_vec_numpy[None, :]
        dists_center_to_center = np.linalg.norm(delta_pos, axis=1)
        combined_obs = combined_obs[dists_center_to_center < cbf_active_range]

        if combined_obs.size == 0:
            return combined_obs
        if max_active_spheres > 0 and combined_obs.shape[0] > max_active_spheres:
            current_delta_pos = combined_obs[:, :3] - current_drone_p_vec_numpy[None, :]
            current_dists_center_to_center = np.linalg.norm(current_delta_pos, axis=1)
            
            sorted_indices = np.argsort(current_dists_center_to_center)
            combined_obs = combined_obs[sorted_indices[:max_active_spheres]]
            
        return combined_obs
