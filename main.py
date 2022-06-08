from algorithm1_auer import auer_algorithm1
from robust_moo import robust_moo

from arm_distribution_generator import *
from find_alpha_suboptimal_arms import find_alpha_suboptimal_and_pareto_llvm, \
                                    find_alpha_suboptimal_and_pareto_diabetes, find_alpha_suboptimal_and_pareto_synthetic

from unavoidable_bias_calculator import UnavoidableBias
from find_alpha_suboptimal_arms import check_covering

from create_table import latex_table
import os



setting = 'synthetic'  #experiment setup
# setting = 'llvm'
# setting = 'diabetes'



latex_code_output_file_name = os.path.join('Results', f'latex_code_{setting}.txt')

dist_name = 'Gaussian' #reward distribution type
advers = 'oblivious' #adversary type

N = 10 #num. of independent runs

if setting == 'synthetic':
    K = 10
    M = 2
    std = 0.1
    reward_max = 10
    reward_min = 0

    delta = 0.1
    alpha = 0.1
    eps_max = 0.4
    eps_min = 0.0
    leng_epsilon_list = 6

elif setting == 'llvm':
    file_name  = 'llvm_dict.pickle' #contains llvm data as described in the paper
    K = 16 #number of arms
    M = 2 #number of objectives
    std= 0.2 #Gaussian reward dist. std
    delta = 0.1 #conf.
    alpha = 0.1 #acc.
    eps_max = 0.4 #max attack prob. tested
    eps_min = 0.0 #min attack prob.
    leng_epsilon_list = 6

elif setting == 'diabetes':
    file_name = 'diabetes_dict.pickle'
    K = 11
    M = 2
    std = 0.1
    alpha = 0.1

    delta = 0.1
    eps_max = 0.4
    eps_min = 0.0
    leng_epsilon_list = 6
    scale_method = 'standardize' #normalize diabetes data
    # scale_method == 'no_standardization'

epsilon0= alpha #Auer epsilon0 is same as accuracy
# epsilon_list = np.linspace(eps_min, eps_max, leng_epsilon_list)
epsilon_list = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
N_eps= len(epsilon_list)


total_sample_matrix_auer = np.zeros([N_eps, N]) #sample complexity
total_sample_matrix_robust = np.zeros([N_eps, N])
correct_pred_matrix_auer = np.zeros([N_eps, N]) #succesful prediction
correct_pred_matrix_robust = np.zeros([N_eps, N])

ratio_of_opt_pred_to_tot_pred_matrix_auer= np.zeros([N_eps, N])  #ratio of optimal arms returned
ratio_of_opt_pred_to_tot_pred_matrix_robust= np.zeros([N_eps, N])

pred_arms_violate_sc2_matrix_auer= np.zeros([N_eps, N]) #predictions that violate coverage condition
pred_arms_violate_sc2_matrix_robust= np.zeros([N_eps, N])

t_bar= 0.49
R = lambda t: std*np.sqrt(2)*(np.sqrt(np.log(1/(1/2-t)))+np.sqrt(np.log(2)))

if setting == 'llvm':
    arms= ArmGenerator(K, M, dist_name)
    arms.load_llvm(file_name) #load llvm data

elif setting == 'synthetic':
    arms_list= []
    for i in range(N):
        arms = ArmGenerator(K, M, dist_name)
        arms.create_medians_2obj(reward_min, reward_max) #for syntetic setting, create reward median
        arms_list.append(arms)

elif setting == 'diabetes':
    arms= ArmGenerator(K, M, dist_name) 
    arms.load_diabetes(file_name, scale_method = scale_method) #load diabetes data

#in the case of syntetic setup, create new median reward vectors at each run
if setting== 'synthetic':
    ind_suboptimal_matrix= np.empty([N, leng_epsilon_list])
    Pareto_ind_list= []
    non_pareto_ind_list  = []
    for i in range(N):

        Pareto_ind, non_pareto_ind = pareto(arms_list[i].median_matrix, K) #optimal and suboptimal indeces
        Pareto_ind_list.append(Pareto_ind)
        non_pareto_ind_list.append(non_pareto_ind)


ind_suboptimal_list_of_lists= list()

for z, epsilon in enumerate(epsilon_list):
    print()
    print()
    print('epsilon', epsilon)
    bias = UnavoidableBias(R, epsilon, advers)
    D = bias.return_D()

    if (setting=='synthetic' or setting == 'llvm') and epsilon == 0.0:
        D = 0

    if setting=='synthetic':
        ind_suboptimal_list= list()
        for k in range(N):
            ind_suboptimal, _, _ = find_alpha_suboptimal_and_pareto_synthetic(arms_list[k].median_matrix, D, K, alpha)
            ind_suboptimal_list.append(ind_suboptimal)
        ind_suboptimal_list_of_lists.append(ind_suboptimal_list)
    elif setting == 'llvm':
        ind_suboptimal, Pareto_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_llvm(arms.y, arms.sample_inds_dict,
                                               D, K, alpha) #could be given outside the for loop since rewards are fixed
    elif setting == 'diabetes':
        ind_suboptimal, Pareto_ind, non_pareto_ind = find_alpha_suboptimal_and_pareto_diabetes(arms.y,
                                                                                               D, K,
                                                                                               alpha)

    total_samples_auer = np.zeros([N, ])
    total_samples_robust = np.zeros([N, ])
    correct_pred_auer = np.zeros([N, ])
    correct_pred_robust = np.zeros([N, ])

    ratio_of_opt_pred_to_tot_pred_auer= np.zeros([N, ])
    ratio_of_opt_pred_to_tot_pred_robust= np.zeros([N, ])
    pred_arms_violate_sc2_auer= np.zeros([N, ])
    pred_arms_violate_sc2_robust= np.zeros([N, ])


    for i in range(N):
        print()
        print('iter', i+ 1)
        if setting == 'synthetic':
            arms= arms_list[i]
            ind_suboptimal= ind_suboptimal_list[i]
            Pareto_ind= Pareto_ind_list[i]
            non_pareto_ind= non_pareto_ind_list[i]

        P_auer, eliminated_auer, P_auer_ind= auer_algorithm1(K, M, epsilon, delta, epsilon0, arms, setting, Pareto_ind, std, D)



        P_robust, eliminated_robust, P_robust_ind = robust_moo(advers, t_bar, K, M, R, epsilon, alpha, delta,
                                                               arms, setting, Pareto_ind, std, D)

        #Auer
        #check accuracy
        suboptimal_auer= True
        for k in P_auer_ind:
            if not k in ind_suboptimal:
                suboptimal_auer= False

        #check covering
        covering_auer= True
        covering_auer= check_covering(arms.median_matrix, D, alpha , P_auer_ind, Pareto_ind)

        if suboptimal_auer and covering_auer:
            correct_pred_auer[i]= True

        #RPSI
        suboptimal_robust= True
        for k in P_robust_ind:
            if not k in ind_suboptimal:
                suboptimal_robust= False
        covering_robust= True
        covering_robust= check_covering(arms.median_matrix, D, alpha , P_robust_ind, Pareto_ind)

        if suboptimal_robust and covering_robust:
            correct_pred_robust[i]= True

        #Auer total samp
        auer_samp= 0
        auer_dict= {**P_auer, **eliminated_auer}
        for arm in auer_dict:
            auer_samp+= auer_dict[arm]['ti']

        total_samples_auer[i] = auer_samp

        #RPSI total samp
        robust_samp= 0
        robust_dict= {**P_robust, **eliminated_robust}
        for arm in robust_dict:
            robust_samp+= robust_dict[arm]['Ni']

        total_samples_robust[i]= robust_samp


        num_pareto_in_pred_auer= 0
        for ind in P_auer_ind:
            if ind in Pareto_ind:
                num_pareto_in_pred_auer+= 1
        ratio_of_opt_pred_to_tot_pred_auer[i] = num_pareto_in_pred_auer/len(Pareto_ind)

        num_pareto_in_pred_robust= 0
        for ind in P_robust_ind:
            if ind in Pareto_ind:
                num_pareto_in_pred_robust+= 1
        ratio_of_opt_pred_to_tot_pred_robust[i] = num_pareto_in_pred_robust/len(Pareto_ind)


        num_violate_sc2_auer= 0
        for ind in P_auer_ind:
            if ind in non_pareto_ind:
                if ind not in ind_suboptimal:
                    num_violate_sc2_auer+=1
        pred_arms_violate_sc2_auer[i] = num_violate_sc2_auer

        num_violate_sc2_robust= 0
        for ind in P_robust_ind:
            if ind in non_pareto_ind:
                if ind not in ind_suboptimal:
                    num_violate_sc2_robust+=1
        pred_arms_violate_sc2_robust[i] = num_violate_sc2_robust



    total_sample_matrix_auer[z, :] = total_samples_auer[:]
    total_sample_matrix_robust[z, :] = total_samples_robust[:]
    correct_pred_matrix_auer[z, :] = correct_pred_auer[:]
    correct_pred_matrix_robust[z, :] = correct_pred_robust[:]


    ratio_of_opt_pred_to_tot_pred_matrix_auer[z, :]= ratio_of_opt_pred_to_tot_pred_auer[:]
    ratio_of_opt_pred_to_tot_pred_matrix_robust[z, :]= ratio_of_opt_pred_to_tot_pred_robust[:]
    pred_arms_violate_sc2_matrix_auer[z, :]  = pred_arms_violate_sc2_auer[:]
    pred_arms_violate_sc2_matrix_robust[z, :]  = pred_arms_violate_sc2_robust[:]
    print()
    print('auer correct pred.:', np.mean(correct_pred_auer), ',  auer total samp.:', np.mean(total_samples_auer))
    print('robust correct pred.:', np.mean(correct_pred_robust), ',   robuts total samp.:', np.mean(total_samples_robust))

total_samp_mean_auer= np.mean(total_sample_matrix_auer, axis = 1)
total_samp_mean_robust= np.mean(total_sample_matrix_robust, axis= 1)
total_samp_std_auer= np.std(total_sample_matrix_auer, axis= 1)
total_samp_std_robust= np.std(total_sample_matrix_robust, axis= 1)

correct_pred_mean_auer= np.mean(correct_pred_matrix_auer, axis= 1)
correct_pred_std_auer= np.std(correct_pred_matrix_robust, axis= 1)
correct_pred_mean_robust= np.mean(correct_pred_matrix_robust, axis= 1)
correct_pred_std_robust= np.std(correct_pred_matrix_robust, axis= 1)

ratio_of_opt_pred_to_tot_pred_mean_auer= np.mean(ratio_of_opt_pred_to_tot_pred_matrix_auer, axis= 1)
ratio_of_opt_pred_to_tot_pred_mean_robust= np.mean(ratio_of_opt_pred_to_tot_pred_matrix_robust, axis= 1)
pred_arms_violate_sc2_mean_auer= np.mean(pred_arms_violate_sc2_matrix_auer, axis= 1)
pred_arms_violate_sc2_mean_robust= np.mean(pred_arms_violate_sc2_matrix_robust, axis= 1)

print()
print()
print('samp. auer:', total_samp_mean_auer)
print('samp. robust:', total_samp_mean_robust)
print()
print('correct pred. auer:', correct_pred_mean_auer)
print('correct pred. robust:', correct_pred_mean_robust)
print()
print()
print()
os.makedirs(os.path.join('Results'), exist_ok= True)

latex_table(epsilon_list, leng_epsilon_list, latex_code_output_file_name,
            correct_pred_mean_auer, correct_pred_mean_robust,
            total_samp_mean_auer, total_samp_mean_robust,
            ratio_of_opt_pred_to_tot_pred_mean_auer,  ratio_of_opt_pred_to_tot_pred_mean_robust,
            pred_arms_violate_sc2_mean_auer, pred_arms_violate_sc2_mean_robust)

