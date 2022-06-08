import numpy as np

def m1_hat(empirical_median_veci, empirical_median_vecj):
    gap= empirical_median_veci- empirical_median_vecj
    if  np.all(gap <=0):
        return np.abs(np.max(empirical_median_veci- empirical_median_vecj))
    else:
        return 0

def get_a1(arm_dict, M):
    include_ind = np.zeros([0, ])
    for arm1 in arm_dict:
        empirical_mean_vec= np.zeros([1, M])
        include= True
        beta1 = arm_dict[arm1]['Beta']
        for i in range(M):
            empirical_mean_vec[0, i]= arm_dict[arm1]['mi_hat'][i]

        for arm2 in arm_dict:
            empirical_mean_vec2 = np.zeros([1, M])
            beta2= arm_dict[arm2]['Beta']

            for k in range(M):
                empirical_mean_vec2[0, k] = arm_dict[arm2]['mi_hat'][k]
            if m1_hat(empirical_mean_vec, empirical_mean_vec2) > beta1+ beta2:
                include= False
                break
        if  include:
            include_ind= np.append(include_ind, np.array([int(arm1)]), axis=0)
    return include_ind

