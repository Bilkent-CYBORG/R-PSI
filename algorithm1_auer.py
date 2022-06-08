import numpy as np
from get_a1 import get_a1
from get_p1 import get_p1
from get_p2 import get_p2
from find_alpha_suboptimal_arms import find_alpha_suboptimal_and_pareto_llvm,\
                                    find_alpha_suboptimal_and_pareto_diabetes, find_alpha_suboptimal_and_pareto_synthetic


def auer_algorithm1(K, M, epsilon, delta, epsilon0, arms, setting, Pareto_ind, std, D):

    eliminated = dict()
    arm_dict= dict()
    P= dict()
    for i in range(K):
        arm_dict[str(i)] = dict()
        arm_dict[str(i)]['ti'] = 0
        arm_dict[str(i)]['Beta'] = 0
        arm_dict[str(i)]['mi_hat'] = np.zeros([M, ])
        arm_dict[str(i)]['samples'] = dict()

        for obj in range(M):
            arm_dict[str(i)]['samples'][obj] = np.zeros([0, ])

    while True:
        for arm in arm_dict:
            arm_dict[arm]['ti'] += 1
            ni = arm_dict[arm]['ti']
            arm_dict[arm]['Beta'] = np.sqrt(2 * np.log(4 * K * M * ni ** 2 / delta) / ni)

            for i in range(M):
                if setting == 'synthetic':
                    arm_dict[arm]['samples'][i] = \
                        np.append(arm_dict[arm]['samples'][i], arms.create_samples(int(arm), i, 1, epsilon, Pareto_ind, std), axis=0) #sample arm
                elif setting == 'llvm':
                    arm_dict[arm]['samples'][i] = \
                        np.append(arm_dict[arm]['samples'][i], arms.create_samples_llvm(int(arm), i, 1, epsilon, Pareto_ind), axis=0)

                elif setting == 'diabetes':
                    arm_dict[arm]['samples'][i] = \
                        np.append(arm_dict[arm]['samples'][i], arms.create_samples_diabetes(int(arm), i, 1, epsilon, std, Pareto_ind), axis=0)

                arm_dict[arm]['mi_hat'][i] = np.mean(arm_dict[arm]['samples'][i])


        A1_ind= get_a1(arm_dict, M)
        P1_ind= get_p1(arm_dict, A1_ind, epsilon0, M)
        P2_ind = get_p2(arm_dict, P1_ind, A1_ind, epsilon0 , M)

        for i in P2_ind:
            P[str(int(i))] = arm_dict.pop(str(int(i)))

        arm_dict_new= dict()

        for arm in arm_dict:
            if int(arm) not in A1_ind:
                eliminated[arm] = arm_dict[arm]
        for i in A1_ind:
            if i in P2_ind:
                pass
            else:
                arm_dict_new[str(int(i))]= arm_dict[str(int(i))]
        arm_dict= arm_dict_new.copy()
        if len(arm_dict.keys()) == 0:
            break

    P_ind = np.zeros([0,])
    for arm in P:
        P_ind= np.append(P_ind, np.array([int(arm)]), axis= 0)
    return P, eliminated, P_ind

