import numpy as np
from maximum_uncertainty_arm import find_arm_with_maximum_uncertainty
from arm_distribution_generator import find_pessimistic_and_eliminate
from get_O1_set import identify_optimal_arms
from get_O2_set import identify_O2_set
from find_alpha_suboptimal_arms import find_alpha_suboptimal_and_pareto_llvm, \
                                        find_alpha_suboptimal_and_pareto_diabetes,find_alpha_suboptimal_and_pareto_synthetic


def robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta, arms, setting, Pareto_ind, std, D):

    arm_dict = dict()

    for i in range(K):
        arm_dict[str(i)] = dict()
        arm_dict[str(i)]['ti'] = 0
        arm_dict[str(i)]['Ui'] = 0
        arm_dict[str(i)]['Ni']= 0
        arm_dict[str(i)]['mi_hat'] = np.zeros([M, ])
        arm_dict[str(i)]['samples']  = dict()

        for obj in range(M):
            arm_dict[str(i)]['samples'][obj]  = np.zeros([0, ])

    if advers == 'prescient' or 'oblivious':
        beta = np.power(t_bar - epsilon/(2*(1-epsilon)), -2)
        h_eps= epsilon/(2* (1-epsilon))
        delta_tild = delta/2
    elif advers == 'malicious':
        beta = np.power((t_bar - epsilon), -2)
        h_eps= epsilon
        delta_tild=delta/3
    else:
        print('incorrect adversary name')


    N_init = np.ceil(2 * beta * np.log((np.pi**2*M*K)/(6 * delta_tild))) #n0

    t = 0

    for arm in arm_dict:
        arm_dict[arm]['ti'] +=1
        ti = arm_dict[arm]['ti']
        arm_dict[arm]['Ni'] += N_init

        arm_dict[arm]['Ui'] = R(h_eps + 1/np.sqrt(beta* ti)) - R(h_eps)


        for i in range(M):
            if setting == 'synthetic':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples(int(arm), i, int(N_init), epsilon, Pareto_ind, std), axis= 0) #sample
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])
            elif setting == 'llvm':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples_llvm(int(arm), i, int(N_init), epsilon, Pareto_ind), axis= 0)
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])
            elif setting == 'diabetes':
                arm_dict[arm]['samples'][i]= \
                    np.append(arm_dict[arm]['samples'][i], arms.create_samples_diabetes(int(arm), i, int(N_init), epsilon, std, Pareto_ind), axis= 0)
                arm_dict[arm]['mi_hat'][i] = np.median( arm_dict[arm]['samples'][i])

    P= dict()
    eliminated= dict()
    while True:
        if t> 0:
            _, arm_max= find_arm_with_maximum_uncertainty(arm_dict)
            arm_dict[arm_max]['ti'] += 1
            ti = arm_dict[arm_max]['ti']

            delt_Ni = np.ceil(1 + 4* ti * beta * np.log(ti/ (ti-1)) + 2 * beta * np.log((ti - 1 ) ** 2 * M * K * np.pi ** 2 / (6 * delta_tild)))

            arm_dict[arm_max]['Ni'] += delt_Ni

            arm_dict[arm_max]['Ui'] = R(h_eps + 1/np.sqrt(beta* ti)) - R(h_eps)

            for i in range(M):
                if setting == 'synthetic':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples(int(arm_max), i, int(delt_Ni), epsilon, Pareto_ind, std), axis= 0)
                elif setting == 'llvm':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples_llvm(int(arm_max), i, int(delt_Ni), epsilon, Pareto_ind), axis= 0)
                elif setting == 'diabetes':
                    arm_dict[arm_max]['samples'][i]= np.append(arm_dict[arm_max]['samples'][i],
                                                                arms.create_samples_diabetes(int(arm_max), i, int(delt_Ni), epsilon, std, Pareto_ind), axis= 0)

                arm_dict[arm_max]['mi_hat'][i] = np.median( arm_dict[arm_max]['samples'][i])

        ind_eliminated= find_pessimistic_and_eliminate(arm_dict, M, D) #elimination step

        if ind_eliminated is None:
            pass
        else:
            for ind in ind_eliminated:
                eliminated[str(int(ind))]= arm_dict.pop(str(int(ind)))

        if len(arm_dict.keys()) == 0:
            break


        O1_index, U_vec= identify_optimal_arms(arm_dict, alpha, M) #Find O1 set

        if np.any(U_vec > alpha/4):
            if not O1_index is None:
                O2_ind= identify_O2_set(O1_index, arm_dict, alpha, M) #Find O2 set
                if O2_ind is not None:
                    for j in O2_ind:

                        P[str(int(j))]= arm_dict.pop(str(int(j)))

                    if len(arm_dict.keys()) == 0:
                        break
        else:
            if not O1_index is None:
                for j in O1_index:
                    P[str(int(j))] = arm_dict.pop(str(int(j)))

            arm_index = list()
            for arm in arm_dict:
                arm_index.append(int(arm))

            for arm_ind in arm_index:
                eliminated[str(int(arm_ind))] = arm_dict.pop(str(int(arm_ind)))
            break

        t += 1

    P_ind= np.zeros([0, ])
    for arm in P:
        P_ind= np.append(P_ind, np.array([int(arm)]), axis= 0)
    return P, eliminated, P_ind

