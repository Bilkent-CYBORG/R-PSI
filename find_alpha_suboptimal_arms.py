import numpy as np
from arm_distribution_generator import pareto


def find_alpha_suboptimal_and_pareto(median_matrix, D, K, alpha):
    pess_ind, non_pess_ind= pareto( median_matrix, K)
    two_D_plus_alpha_suboptimal_ind= np.zeros([0,])
    for j in non_pess_ind:
        median_vec= median_matrix[j]
        two_D_plus_alpha_suboptimal= True
        for i in pess_ind:
            median_vec_pess= median_matrix[i]
            if np.all(median_vec + 2 * D + alpha<= median_vec_pess):
                two_D_plus_alpha_suboptimal = False
                break
        if two_D_plus_alpha_suboptimal:

            two_D_plus_alpha_suboptimal_ind = np.append(two_D_plus_alpha_suboptimal_ind, np.array([j]), axis= 0)
    return np.append(pess_ind, two_D_plus_alpha_suboptimal_ind,  axis= 0)

def find_alpha_suboptimal_and_pareto_llvm(y, sample_inds_dict, D,  K, alpha):
    arm_means= np.zeros([K, 2])
    for arm in sample_inds_dict:
        sample_inds = sample_inds_dict[arm]
        arm_means[arm, :]=  np.mean(y[sample_inds], axis= 0)
    pareto_ind, non_pareto_ind= pareto(arm_means, 16)
    alpha_suboptimal_ind = np.zeros([0, ])
    for j in non_pareto_ind:
        mean_vec= arm_means[j]
        alpha_suboptimal = True

        for i in pareto_ind:
            mean_vec_pareto = arm_means[i]
            if np.all(mean_vec + 2 * D + alpha <= mean_vec_pareto):
                alpha_suboptimal = False
                break

        if alpha_suboptimal:
            alpha_suboptimal_ind = np.append(alpha_suboptimal_ind, np.array([j]), axis= 0)

    return np.append(pareto_ind, alpha_suboptimal_ind,  axis= 0), pareto_ind, non_pareto_ind

def find_alpha_suboptimal_and_pareto_synthetic(y, D, K, alpha):
    ind_suboptimal = find_alpha_suboptimal_and_pareto(y, D, K, alpha)
    pareto_ind, non_pareto_ind = pareto(y, K)
    return ind_suboptimal, pareto_ind, non_pareto_ind

def find_alpha_suboptimal_and_pareto_diabetes(y, D, K, alpha):
    ind_suboptimal = find_alpha_suboptimal_and_pareto(y, D, K, alpha)
    pareto_ind, non_pareto_ind = pareto(y, K)
    return ind_suboptimal, pareto_ind, non_pareto_ind

def check_covering(median_matrix, D, alpha, P, P_true):
    all_covered = True
    for p_ind in P_true:
        median_vec_p = median_matrix[int(p_ind)]
        covered = False
        for p_pred in P:
            median_vec_p_pred = median_matrix[int(p_pred)]
            if np.all(median_vec_p_pred + 2* D + alpha >= median_vec_p):
                covered = True
                break
        if covered == False:
            all_covered= False
            break

    return all_covered
