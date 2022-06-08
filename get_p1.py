import numpy as np

def M_hat_epsilon0(emp_median_vec1, emp_median_vec2, epsilon0):
    gap= (emp_median_vec1+ epsilon0) - (emp_median_vec2)
    if np.all(gap <= 0):
        return 0
    else:
        return np.max(gap)

def get_p1(arm_dict, A1_ind, epsilon0, M):
    A1= dict()
    for j in A1_ind:
        A1[str(int(j))] = arm_dict[str(int(j))]
    optimal_ind= np.zeros([0,])
    for arm1 in A1:
        optimal= True
        empirical_median_vec1= np.zeros([1, M])
        beta1 = A1[arm1]['Beta']
        for i in range(M):
            empirical_median_vec1[0, i] = A1[arm1]['mi_hat'][i]

        for arm2 in A1:

            empirical_median_vec2= np.zeros([1, M])
            beta2 = A1[arm2]['Beta']
            for j in range(M):
                empirical_median_vec2[0, j]= A1[arm2]['mi_hat'][j]

            if arm1 == arm2:
                pass
            else:
                if M_hat_epsilon0(empirical_median_vec1, empirical_median_vec2, epsilon0) < beta1 + beta2:
                    optimal =False
                    break
        if optimal == True:
            optimal_ind= np.append(optimal_ind, np.array([int(arm1)]), axis = 0)

    return optimal_ind