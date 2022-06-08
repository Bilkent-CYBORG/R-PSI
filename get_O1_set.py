import numpy as np
def identify_optimal_arms(arm_dict, alpha, M):
    empirical_median_matrix= np.zeros([0, M])
    U_vec= np.zeros([0, 1])
    existing_arm_index= np.zeros([0,])
    for arm in arm_dict:
        empirical_median_vec= np.zeros([1, M])
        U= arm_dict[arm]['Ui']
        for j in range(M):

            empirical_median_vec[0, j] =arm_dict[arm]['mi_hat'][j]
        U_vec= np.append(U_vec, np.array([[U]]), axis=  0)

        empirical_median_matrix = np.append(empirical_median_matrix, empirical_median_vec, axis= 0)
        existing_arm_index= np.append(existing_arm_index, np.array([int(arm)]), axis=0)

    optimal= np.ones([len(existing_arm_index), 1])
    for i in range(empirical_median_matrix.shape[0]):
        median_vec1=  empirical_median_matrix[i]
        U1= U_vec[i, 0]
        for j in range(i+1, empirical_median_matrix.shape[0]):
            median_vec2= empirical_median_matrix[j]
            U2= U_vec[j, 0]

            if np.all(median_vec1 - U1 +alpha <= median_vec2 + U2):
                optimal[i, 0] = 0
            if np.all(median_vec2- U2 + alpha<= median_vec1 + U1):
                optimal[j, 0] = 0
    if len(optimal.nonzero()[0]) == 0:
        return None, U_vec
    else:
        return existing_arm_index[optimal.nonzero()[0]], U_vec
