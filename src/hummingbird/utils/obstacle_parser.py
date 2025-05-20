import ast
import numpy as np
import rospy

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