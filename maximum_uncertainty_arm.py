import numpy as np

def find_arm_with_maximum_uncertainty(arm_dict):
    t_opt= np.inf
    for arm in arm_dict:
        if arm_dict[arm]['ti'] < t_opt:
            t_opt= arm_dict[arm]['ti']
            i_opt= arm

    return t_opt, i_opt