import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.patches as mpatches
from util import pickle_read
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler


# find pareto arms from reward vectors
def pareto(matrix, K):
    non_dominated= np.ones([K])
    dominated= np.zeros([K])
    for i in range(K):
        for j in range(i+1, K):
            if np.all(matrix[i] <= matrix[j]):
                non_dominated[i] =  0
                dominated[i] = 1
            elif np.all(matrix[j] <= matrix[i]):
                non_dominated[j] = 0
                dominated[j]= 1
    return non_dominated.nonzero()[0], dominated.nonzero()[0]


class ArmGenerator:
    def __init__(self, K, M, dist_name):
        self.K= K
        self.dist_name= dist_name
        self.M= M

    def create_medians_2obj(self, reward_min, reward_max):
        K, M = self.K, self.M
        median_matrix= np.random.uniform(reward_min,reward_max,(K, M))
        self.median_matrix  = median_matrix


    def create_samples(self, arm_ind, obj_ind, N, epsilon, pareto_inds,std):
        indicator = np.random.binomial(1, epsilon, size= N)
        median_matrix= self.median_matrix
        # contam_amp = 1000
        contam_amp = 1
        if arm_ind in pareto_inds:
            contamination= -contam_amp
        else:
            contamination = contam_amp

        if self.dist_name == 'Gaussian':
            samples= np.random.normal(median_matrix[arm_ind][obj_ind], size= N, scale= std) \
                     * ( 1-indicator) + indicator * contamination
        return samples

    def create_samples_diabetes(self, arm_ind, obj_ind, N, epsilon, sigma, pareto_inds):
        indicator = np.random.binomial(1, epsilon, size= N)
        median_matrix= self.y
        contam_amp = np.random.uniform(-50, 50)
        if arm_ind in pareto_inds:
            contamination= -contam_amp
        else:
            contamination = contam_amp

        if self.dist_name == 'Gaussian':
            samples= np.random.normal(median_matrix[arm_ind][obj_ind], size= N, scale= sigma) \
                     + indicator * contamination
        return samples



    def create_samples_llvm(self, arm_ind, obj_ind, N, epsilon, pareto_inds):
        # contam_amp = 1000
        # contam_amp = 1
        contam_amp = 10
        if arm_ind in pareto_inds:
            contamination= -contam_amp
        else:
            contamination = contam_amp
        indicator = np.random.binomial(1, epsilon, size= N)
        sample_inds= self.sample_inds_dict[arm_ind]
        random_sample_inds= sample_inds[np.random.choice(len(sample_inds), size= N, replace= True)]
        true_samples =self.y[random_sample_inds][:, obj_ind]
        corrupted_samples = true_samples * (1- indicator) + indicator* contamination

        return corrupted_samples


    def load_llvm(self, file_name):
        llvm_dict= pickle_read(file_name)
        self.y = llvm_dict['y']
        self.x= llvm_dict['x']
        self.sample_inds_dict= llvm_dict['sample_inds_dict']

        median_matrix= np.zeros([0,2])
        for arm in self.sample_inds_dict:
            mean_arm = np.mean(self.y[self.sample_inds_dict[arm], :], axis= 0, keepdims= True)
            median_matrix = np.append(median_matrix, mean_arm, axis = 0)

        self.median_matrix = median_matrix

        std_matrix = np.zeros([0,2])
        for arm in self.sample_inds_dict:
            std_arm = np.std(self.y[self.sample_inds_dict[arm], :], axis= 0, keepdims= True)
            std_matrix = np.append(std_matrix, std_arm, axis = 0)
        self.std_matrix= std_matrix

    def load_diabetes(self, file_name, scale_method):

        if scale_method== 'standardize':
            diabetes_dict = pickle_read(file_name)

            scaler = StandardScaler()
            data= diabetes_dict['y']
            scaler.fit(data)
            data_stand= scaler.transform(data)

            self.y = data_stand
            self.x= diabetes_dict['x']
            self.median_matrix  = self.y

        elif scale_method== 'no_standardization':
            diabetes_dict = pickle_read(file_name)
            self.y =diabetes_dict['y']
            self.x= diabetes_dict['x']
            self.median_matrix  = self.y



def find_pessimistic_and_eliminate(arm_dict, M, D):
    empirical_median_matrix = np.zeros([0, M])
    existing_arm_index = np.zeros([0, ])
    U_vec= np.zeros([0, ])
    for arm in arm_dict:
        empirical_median_vec= np.zeros([1, M])
        U= arm_dict[arm]['Ui']
        U_vec= np.append(U_vec, [U], axis= 0)
        for j in range(M):
            empirical_median_vec[0, j] = arm_dict[arm]['mi_hat'][j]
        empirical_median_matrix= np.append(empirical_median_matrix, empirical_median_vec, axis= 0)
        existing_arm_index= np.append(existing_arm_index, int(arm))

    lower_confidence=  empirical_median_matrix- np.expand_dims(U_vec, axis=1) - D
    pess_ind, non_pess_ind= pareto(lower_confidence, K= lower_confidence.shape[0])

    dominated= np.zeros([len(existing_arm_index), 1])
    non_dominated= np.ones([len(existing_arm_index), 1])
    for i, arm in enumerate(existing_arm_index):
        median_vec= empirical_median_matrix[i]
        U= U_vec[i]
        for _pess_ind in pess_ind:
            median_vec_pess= empirical_median_matrix[_pess_ind]
            U_pess= U_vec[_pess_ind]
            if _pess_ind == i:
                pass
            elif np.all(median_vec_pess- U_pess -D>= (median_vec + D +U)):
                dominated[i, 0] = 1
                non_dominated[i, 0] = 0

    if len(dominated.nonzero()[0]) == 0:
        return None
    else:
        return existing_arm_index[dominated.nonzero()[0]]
