import numpy as np

def identify_O2_set(O1_index, arm_dict, alpha, M):
    O2_ind= np.ones([len(O1_index),1 ])
    for k, ind in enumerate(O1_index):
        empirical_median_vec= np.zeros([1, M])
        U= arm_dict[str(int(ind))]['Ui']
        for j in range(M):
            empirical_median_vec[0, j ] = arm_dict[str(int(ind))]['mi_hat'][j]

        for arm in arm_dict:
            if int(arm) == int(ind):
                pass
            else:
                empirical_median_vec2= np.zeros([1, M])
                U2= arm_dict[arm]['Ui']

                for j in range(M):
                    empirical_median_vec2[0, j]= arm_dict[arm]['mi_hat'][j]

                if np.all(empirical_median_vec2 - U2 + alpha <= empirical_median_vec+ U):
                    O2_ind[k]= 0
    if len(O2_ind.nonzero()[0]) == 0:
        return None
    else:
        return O1_index[O2_ind.nonzero()[0]]