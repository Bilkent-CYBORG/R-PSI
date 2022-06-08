import numpy as np
from get_p1 import M_hat_epsilon0

def get_p2(arm_dict, P1_ind, A1_ind, epsilon0 , M):
    P1= dict()
    A1= dict()
    P2_ind= np.zeros([0, ])
    for i in P1_ind:
        P1[str(int(i))] = arm_dict[str(int(i))]
    for j in A1_ind:
        A1[str(int(j))]= arm_dict[str(int(j))]
    for arm1 in P1:
        empirical_median_vec1= np.zeros([1,M])
        Beta1= P1[arm1]['Beta']

        for i in range(M):
            empirical_median_vec1[0, i] = P1[arm1]['mi_hat'][i]
        optimal = True
        for arm2 in A1:
            if arm2 in P1:
                pass
            else:
                empirical_median_vec2= np.zeros([1, M])
                Beta2= A1[arm2]['Beta']
                for i in range(M):
                    empirical_median_vec2[0, i]= A1[arm2]['mi_hat'][i]

                if M_hat_epsilon0(empirical_median_vec2, empirical_median_vec1, epsilon0) <= Beta1 + Beta2:
                    optimal =False
                    break
        if optimal == True:
            P2_ind= np.append(P2_ind, np.array([int(arm1)]), axis= 0)
    return P2_ind
