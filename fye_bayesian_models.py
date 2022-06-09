import csv

import numpy as np
import matplotlib.pyplot as plt

from bayesian_tools.LM_Core import LM_base, Regression_Model_Checker, InfomationCriteria_for_LM
from bayesian_tools.MCMC_Core import MCMC_Diag
from special_dist_sampler.sampler_gamma import Sampler_univariate_InvGamma


class PinesData:
    def __init__(self):
        self._load()
        self._normalize()
        self._coded_vec()

    def _load(self, file_path = "data_pines.csv"):
        self.species=[]
        self.tree_id = []
        self.dfromtop = []
        self.lma = []

        self.tree_id_set_FP = set() # Pinus ponderosa
        self.tree_id_set_FW = set() # Pinus monticola

        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            #header
            # j     i       x_ijk               y_ijk
            # ID	species	dfromtop	height	LMA
            # 0     1       2           3       4
            for row in csv_reader:
                self.tree_id.append(str(row[0]))
                self.species.append(str(row[1]))
                self.dfromtop.append(float(row[2]))
                self.lma.append(float(row[4]))

        self.coded_dict = {}
        i = 0
        for id in self.tree_id:
            if id not in self.coded_dict.keys():
                self.coded_dict[id] = i
                i += 1

        print(self.coded_dict)
        self.num_data = len(self.lma)
        self.num_category = len(self.coded_dict.keys())

    def _normalize(self):
        self.lma_mean = np.mean(self.lma)
        self.lma_sd = np.std(self.lma)
        self.dfromtop_mean = np.mean(self.dfromtop)
        self.dfromtop_sd = np.std(self.dfromtop)
        self.lma = [(x-self.lma_mean)/self.lma_sd for x in self.lma]
        self.dfromtop = [(x-self.dfromtop_mean)/self.dfromtop_sd for x in self.dfromtop]
        print(self.lma_mean, self.lma_sd, self.dfromtop_mean, self.dfromtop_sd)


    def _coded_vec(self):
        self.coded_mat_indicator = np.zeros((self.num_data, self.num_category))
        self.coded_mat_dfromtop = np.zeros((self.num_data, self.num_category))
        self.coded_mat_dfromtop_by_species_only = np.zeros((self.num_data, 2))
        for i, (id, dft) in enumerate(zip(self.tree_id, self.dfromtop)):
            col_idx = self.coded_dict[id]
            self.coded_mat_indicator[i, col_idx] = 1
            self.coded_mat_dfromtop[i, col_idx] = dft
            if id[:2] == "FP":
                self.coded_mat_dfromtop_by_species_only[i, 0] = dft
            else:
                self.coded_mat_dfromtop_by_species_only[i, 1] = dft

    def get_values_for_normalizing(self):
        return (self.lma_mean, self.lma_sd, self.dfromtop_mean, self.dfromtop_sd)

    def get_yX_for_model1(self):
        model1_X = np.c_[self.coded_mat_indicator, self.coded_mat_dfromtop_by_species_only]
        return (np.array(self.lma), model1_X)
    
    def get_yX_for_model2(self):
        model1_X = np.c_[self.coded_mat_indicator, self.coded_mat_dfromtop]
        return (np.array(self.lma), model1_X)




class TH1Model1(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.num_data = len(self.y)

        self.MC_sample = [initial]
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        # able to tune these
        self.hyper_m_10 = 0
        self.hyper_s2_10 = 100
        self.hyper_m_20 = 0
        self.hyper_s2_20 = 100
        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau = 0.01
        self.hyper_b_tau = 0.01
        self.hyper_phi2 = 100


    def full_conditional_sampler_beta_tilde(self, last_param):
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        sigma2 = last_param[7]
        tau2 = last_param[6]
        mu_10 = last_param[4]
        mu_20 = last_param[5]

        m_array = np.array([mu_10 for _ in range(len(last_param[0]))] + [mu_20 for _ in range(len(last_param[1]))] + [0, 0])
        D_inv = np.diag(1 / np.array([tau2]*20 + [self.hyper_phi2]*2))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        
        new_mu_1 = new_coeff_vec[0:8].tolist()
        new_mu_2 = new_coeff_vec[8:20].tolist()
        new_beta1 = new_coeff_vec[20]
        new_beta2 = new_coeff_vec[21]
        new_sample = [new_mu_1, new_mu_2, new_beta1, new_beta2, last_param[4], last_param[5], last_param[6], last_param[7]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        coeff_vec = np.array(last_param[0] + last_param[1] + [last_param[2], last_param[3]])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ coeff_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], new_sigma2]
        return new_sample

    def full_conditional_sampler_mu_10(self, last_param):
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        tau2 = last_param[6]
        mu_1 = last_param[0]
        param_var = 1/(len(mu_1)/tau2 + 1/self.hyper_s2_10)
        param_mean = param_var * (np.sum(mu_1) / tau2 + self.hyper_m_10 / self.hyper_s2_10)
        new_mu_10 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], new_mu_10, last_param[5], last_param[6], last_param[7]]
        return new_sample
    
    def full_conditional_sampler_mu_20(self, last_param):
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        tau2 = last_param[6]
        mu_2 = last_param[1]
        param_var = 1/(len(mu_2)/tau2 + 1/self.hyper_s2_20)
        param_mean = param_var * (np.sum(mu_2) / tau2 + self.hyper_m_20 / self.hyper_s2_20)
        new_mu_20 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        # new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], last_param[7]]
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], new_mu_20, last_param[6], last_param[7]]
        return new_sample


    def full_conditional_sampler_tau2(self, last_param):
        # 0                1                  2      3      4     5     6     7
        #[[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        mu_1 = last_param[0]
        mu_2 = last_param[1]
        param_a = self.hyper_a_tau + (len(mu_1) + len(mu_2)) / 2

        mu_i = np.array(mu_1 + mu_2)
        mu = np.array([last_param[4]]*len(mu_1) + [last_param[5]]*len(mu_2))
        resid = mu_i - mu
        param_b = self.hyper_b_tau + np.dot(resid, resid) / 2
        new_tau2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], new_tau2, last_param[7]]
        return new_sample

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta_tilde(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu_10(new)
        new = self.full_conditional_sampler_mu_20(new)
        new = self.full_conditional_sampler_tau2(new)
        self.MC_sample.append(new)





class TH1Model2(LM_base):
    def __init__(self, response_vec, design_matrix, initial, rnd_seed=None) -> None:
        super().__init__(response_vec, design_matrix, rnd_seed)
        self.np_rng = np.random.default_rng(seed=rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma(set_seed=rnd_seed)

        self.num_data = len(self.y)

        self.MC_sample = [initial]
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        # able to tune these
        self.hyper_m_10 = 0
        self.hyper_s2_10 = 100
        self.hyper_m_20 = 0
        self.hyper_s2_20 = 100
        self.hyper_m_11 = 0
        self.hyper_s2_11 = 100
        self.hyper_m_21 = 0
        self.hyper_s2_21 = 100

        self.hyper_a_sigma = 0.01
        self.hyper_b_sigma = 0.01
        self.hyper_a_tau = 0.01
        self.hyper_b_tau = 0.01
        self.hyper_a_phi = 0.01
        self.hyper_b_phi = 0.01



    def full_conditional_sampler_beta_tilde(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        tau2 = last_param[8]
        phi2 = last_param[9]
        sigma2 = last_param[10]


        m_array = np.array([last_param[4]]*8 + [last_param[5]]*12 + [last_param[6]]*8 + [last_param[7]]*12)
        D_inv = np.diag(1 / np.array([tau2]*20 + [phi2]*20))

        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        
        new_mu_1 = new_coeff_vec[0:8].tolist()
        new_mu_2 = new_coeff_vec[8:20].tolist()
        new_beta_1 = new_coeff_vec[20:28].tolist()
        new_beta_2 = new_coeff_vec[28:40].tolist()
        new_sample = [new_mu_1, new_mu_2, new_beta_1, new_beta_2, last_param[4], last_param[5], last_param[6], last_param[7], last_param[8], last_param[9], last_param[10]]
        return new_sample

    def full_conditional_sampler_sigma2(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        coeff_vec = np.array(last_param[0] + last_param[1] + last_param[2] + last_param[3])
        param_a = self.hyper_a_sigma + self.num_data / 2
        resid = self.y - self.x @ coeff_vec
        param_b = self.hyper_b_sigma + np.dot(resid, resid) / 2
        new_sigma2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], last_param[7], last_param[8], last_param[9], new_sigma2]
        return new_sample

    def full_conditional_sampler_mu_10(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        tau2 = last_param[8]
        mu_1 = last_param[0]
        param_var = 1/(len(mu_1)/tau2 + 1/self.hyper_s2_10)
        param_mean = param_var * (np.sum(mu_1) / tau2 + self.hyper_m_10 / self.hyper_s2_10)
        new_mu_10 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], new_mu_10, last_param[5], last_param[6], last_param[7], last_param[8], last_param[9], last_param[10]]
        return new_sample
    
    def full_conditional_sampler_mu_20(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        tau2 = last_param[8]
        mu_2 = last_param[1]
        param_var = 1/(len(mu_2)/tau2 + 1/self.hyper_s2_20)
        param_mean = param_var * (np.sum(mu_2) / tau2 + self.hyper_m_20 / self.hyper_s2_20)
        new_mu_20 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], new_mu_20, last_param[6], last_param[7], last_param[8], last_param[9], last_param[10]]
        return new_sample

 
    def full_conditional_sampler_gamma1(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        phi2 = last_param[9]
        beta_1 = last_param[2]
        param_var = 1/(len(beta_1)/phi2 + 1/self.hyper_s2_11)
        param_mean = param_var * (np.sum(beta_1) / phi2 + self.hyper_m_11 / self.hyper_s2_11)
        new_gamma1 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], new_gamma1, last_param[7], last_param[8], last_param[9], last_param[10]]
        return new_sample

 
    def full_conditional_sampler_gamma2(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        phi2 = last_param[9]
        beta_2 = last_param[3]
        param_var = 1/(len(beta_2)/phi2 + 1/self.hyper_s2_21)
        param_mean = param_var * (np.sum(beta_2) / phi2 + self.hyper_m_21 / self.hyper_s2_21)
        new_gamma2 = self.np_rng.normal(param_mean, np.sqrt(param_var))
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], new_gamma2, last_param[8], last_param[9], last_param[10]]
        return new_sample


    def full_conditional_sampler_tau2(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        mu_1 = last_param[0]
        mu_2 = last_param[1]
        param_a = self.hyper_a_tau + (len(mu_1) + len(mu_2)) / 2

        mu_i = np.array(mu_1 + mu_2)
        mu = np.array([last_param[4]]*len(mu_1) + [last_param[5]]*len(mu_2))
        resid = mu_i - mu
        param_b = self.hyper_b_tau + np.dot(resid, resid) / 2
        new_tau2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], last_param[7], new_tau2, last_param[9], last_param[10]]
        return new_sample

    def full_conditional_sampler_phi2(self, last_param):
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        beta_1 = last_param[2]
        beta_2 = last_param[3]
        param_a = self.hyper_a_phi + (len(beta_1) + len(beta_2)) / 2

        beta_i = np.array(beta_1 + beta_2)
        gamma = np.array([last_param[6]]*len(beta_1) + [last_param[7]]*len(beta_2))
        resid = beta_i - gamma
        param_b = self.hyper_b_phi + np.dot(resid, resid) / 2
        new_phi2 = self.inv_gamma_sampler.sampler(param_a, param_b)
        new_sample = [last_param[0], last_param[1], last_param[2], last_param[3], last_param[4], last_param[5], last_param[6], last_param[7], last_param[8], new_phi2, last_param[10]]
        return new_sample

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        
        #update new
        new = self.full_conditional_sampler_beta_tilde(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu_10(new)
        new = self.full_conditional_sampler_mu_20(new)
        new = self.full_conditional_sampler_gamma1(new)
        new = self.full_conditional_sampler_gamma2(new)
        new = self.full_conditional_sampler_tau2(new)
        new = self.full_conditional_sampler_phi2(new)
        self.MC_sample.append(new)



if __name__=="__main__":
    factory_inst = PinesData()
    y, x1 = factory_inst.get_yX_for_model1()
    y, x2 = factory_inst.get_yX_for_model2()
    # print(x1.shape)
    # print(x2.shape)
    # print(y.shape)

    model1_run = True
    model2_run = True

    if model1_run:
        # model 1
        # 0                1                  2      3      4     5     6     7
        # [[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        model1_initial = [[0 for _ in range(8)], [0 for _ in range(12)], 0, 0, 0, 0, 1, 1]
        model1_inst = TH1Model1(y, x1, model1_initial, 20220607)
        model1_inst.generate_samples(30000, print_iter_cycle=5000)

        model1_mu_1 = [x[0] for x in model1_inst.MC_sample]
        model1_mu_2 = [x[1] for x in model1_inst.MC_sample]
        model1_others = [x[2:] for x in model1_inst.MC_sample]

        diag_inst11 = MCMC_Diag()
        diag_inst11.set_mc_samples_from_list(model1_mu_1)
        diag_inst11.set_variable_names(["mu1"+str(i) for i in range(1,9)])
        diag_inst11.burnin(3000)
        diag_inst11.show_traceplot((2,4))
        diag_inst11.show_acf(30, (2,4))
        diag_inst11.show_hist_superimposed(y_lab="mu1j")
        diag_inst11.print_summaries(round=3)
        
        diag_inst12 = MCMC_Diag()
        diag_inst12.set_mc_samples_from_list(model1_mu_2)
        diag_inst12.set_variable_names(["mu2"+str(i) for i in range(1,13)])
        diag_inst12.burnin(3000)
        diag_inst12.show_traceplot((3,4))
        diag_inst12.show_acf(30, (3,4))
        diag_inst12.show_hist_superimposed(y_lab="mu2j")
        diag_inst12.print_summaries(round=3)
        
        diag_inst13 = MCMC_Diag()
        diag_inst13.set_mc_samples_from_list(model1_others)
        diag_inst13.set_variable_names(["beta1", "beta2", "mu10", "mu20", "tau2", "sigma2"])
        diag_inst13.burnin(3000)
        diag_inst13.show_traceplot((2,3))
        diag_inst13.show_acf(30, (2,3))
        diag_inst13.show_hist((2,2),[0,1,2,3])
        diag_inst13.show_hist((1,2),[4,5])
        diag_inst13.print_summaries(round=3)

        # 0                1                  2      3      4     5     6     7
        # [[mu11,...,mu18], [mu21,...,mu2.12], beta1, beta2, mu10, mu20, tau2, sigma2]
        model1_coeff = [x[0]+x[1]+[x[2]]+[x[3]] for x in model1_inst.MC_sample]
        model1_coeff = model1_coeff[3000:]
        print("coeff:", model1_coeff[0])
        model1_sigma2 = [x[7] for x in model1_inst.MC_sample]
        model1_sigma2 = model1_sigma2[3000:]

        checker1_inst = Regression_Model_Checker(y, x1, model1_coeff, model1_sigma2)
        checker1_inst.show_posterior_predictive_at_given_data_point(0, show=False, x_lab="")
        checker1_inst.show_posterior_predictive_at_given_data_point(40, show=False, x_lab="")
        checker1_inst.show_posterior_predictive_at_given_data_point(80, show=False, x_lab="")
        checker1_inst.show_posterior_predictive_at_given_data_point(120, show=False, x_lab="")
        checker1_inst.show_posterior_predictive_at_given_data_point(159, show=False, x_lab="")
        plt.show()

        checker1_inst.show_residual_normalProbplot()
        checker1_inst.show_residual_plot()

        nomalize_factor = factory_inst.get_values_for_normalizing()
        normalized_2330dft = (2.330 - nomalize_factor[2]) / nomalize_factor[3]
        normalized_10742dft = (10.742 - nomalize_factor[2]) / nomalize_factor[3]
        print("normalized: ", normalized_2330dft, normalized_10742dft)
        design_row_2330dft = [0 for _ in range(22)]
        design_row_10742dft = [0 for _ in range(22)]
        design_row_2330dft[0] = 1
        design_row_10742dft[0] = 1
        design_row_2330dft[20] = normalized_2330dft
        design_row_10742dft[20] = normalized_10742dft
        design_row_2330dft = np.array(design_row_2330dft)
        design_row_10742dft = np.array(design_row_10742dft)
        checker1_inst.show_posterior_predictive_at_new_point(design_row_2330dft, x_lab="at 2.33")
        checker1_inst.show_posterior_predictive_at_new_point(design_row_10742dft, x_lab="at 10.742")

        ic1_inst = InfomationCriteria_for_LM(y, x1, np.array(model1_coeff), np.array(model1_sigma2))
        print("DIC - model1 :", ic1_inst.DIC())
        print("WAIC - model1 :", ic1_inst.WAIC())


    if model2_run:
        # model 2
        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        model2_initial = [[0 for _ in range(8)], [0 for _ in range(12)], [0 for _ in range(8)], [0 for _ in range(12)], 0, 0, 0, 0, 1, 1, 1]
        model2_inst = TH1Model2(y, x2, model2_initial, 20220607)
        model2_inst.generate_samples(30000, print_iter_cycle=5000)

        model2_mu_1 = [x[0] for x in model2_inst.MC_sample]
        model2_mu_2 = [x[1] for x in model2_inst.MC_sample]
        model2_beta_1 = [x[2] for x in model2_inst.MC_sample]
        model2_beta_2 = [x[3] for x in model2_inst.MC_sample]
        model2_others = [x[4:] for x in model2_inst.MC_sample]

        diag_inst21 = MCMC_Diag()
        diag_inst21.set_mc_samples_from_list(model2_mu_1)
        diag_inst21.set_variable_names(["mu1"+str(i) for i in range(1,9)])
        diag_inst21.burnin(3000)
        diag_inst21.show_traceplot((2,4))
        diag_inst21.show_acf(30, (2,4))
        diag_inst21.show_hist_superimposed(y_lab="mu1j")
        diag_inst21.print_summaries(round=3)

        diag_inst22 = MCMC_Diag()
        diag_inst22.set_mc_samples_from_list(model2_mu_2)
        diag_inst22.set_variable_names(["mu2"+str(i) for i in range(1,13)])
        diag_inst22.burnin(3000)
        diag_inst22.show_traceplot((3,4))
        diag_inst22.show_acf(30, (3,4))
        diag_inst22.show_hist_superimposed(y_lab="mu2j")
        diag_inst22.print_summaries(round=3)
            
        diag_inst23 = MCMC_Diag()
        diag_inst23.set_mc_samples_from_list(model2_beta_1)
        diag_inst23.set_variable_names(["beta1"+str(i) for i in range(1,9)])
        diag_inst23.burnin(3000)
        diag_inst23.show_traceplot((2,4))
        diag_inst23.show_acf(30, (2,4))
        diag_inst23.show_hist_superimposed(y_lab="beta1j")
        diag_inst23.print_summaries(round=3)

        diag_inst24 = MCMC_Diag()
        diag_inst24.set_mc_samples_from_list(model2_beta_2)
        diag_inst24.set_variable_names(["beta2"+str(i) for i in range(1,13)])
        diag_inst24.burnin(3000)
        diag_inst24.show_traceplot((3,4))
        diag_inst24.show_acf(30, (3,4))
        diag_inst24.show_hist_superimposed(y_lab="beta2j")
        diag_inst24.print_summaries(round=3)

        diag_inst25 = MCMC_Diag()
        diag_inst25.set_mc_samples_from_list(model2_others)
        diag_inst25.set_variable_names(["mu10", "mu20", "gamma1", "gamma2", "tau2", "phi2", "sigma2"])
        diag_inst25.burnin(3000)
        diag_inst25.show_traceplot((2,4))
        diag_inst25.show_acf(30, (2,4))
        diag_inst25.show_hist((2,2),[0,1,2,3])
        diag_inst25.show_hist((1,3),[4,5,6])
        diag_inst25.print_summaries(round=3)

        # 0                1                  2                    3                      4     5     6       7       8     9     10
        #[[mu11,...,mu18], [mu21,...,mu2.12], [beta11,...,beta18], [beta21,...,beta2.12], mu10, mu20, gamma1, gamma2, tau2, phi2, sigma2]
        model2_coeff = [x[0]+x[1]+x[2]+x[3] for x in model2_inst.MC_sample]
        model2_coeff = model2_coeff[3000:]
        print("coeff:", model2_coeff[0])
        model2_sigma2 = [x[10] for x in model2_inst.MC_sample]
        model2_sigma2 = model2_sigma2[3000:]

        checker2_inst = Regression_Model_Checker(y, x2, model2_coeff, model2_sigma2)
        checker2_inst.show_posterior_predictive_at_given_data_point(0, show=False, x_lab="")
        checker2_inst.show_posterior_predictive_at_given_data_point(40, show=False, x_lab="")
        checker2_inst.show_posterior_predictive_at_given_data_point(80, show=False, x_lab="")
        checker2_inst.show_posterior_predictive_at_given_data_point(120, show=False, x_lab="")
        checker2_inst.show_posterior_predictive_at_given_data_point(159, show=False, x_lab="")
        plt.show()

        checker2_inst.show_residual_normalProbplot()
        checker2_inst.show_residual_plot()

        nomalize_factor = factory_inst.get_values_for_normalizing()
        normalized_2330dft = (2.330 - nomalize_factor[2]) / nomalize_factor[3]
        normalized_10742dft = (10.742 - nomalize_factor[2]) / nomalize_factor[3]
        print("normalized: ", normalized_2330dft, normalized_10742dft)
        design_row_2330dft = [0 for _ in range(40)]
        design_row_10742dft = [0 for _ in range(40)]
        design_row_2330dft[0] = 1
        design_row_10742dft[0] = 1
        design_row_2330dft[20] = normalized_2330dft
        design_row_10742dft[20] = normalized_10742dft
        design_row_2330dft = np.array(design_row_2330dft)
        design_row_10742dft = np.array(design_row_10742dft)
        checker2_inst.show_posterior_predictive_at_new_point(design_row_2330dft, x_lab="at 2.33")
        checker2_inst.show_posterior_predictive_at_new_point(design_row_10742dft, x_lab="at 10.742")

        ic2_inst = InfomationCriteria_for_LM(y, x2, np.array(model2_coeff), np.array(model2_sigma2))
        print("DIC - model2 :", ic2_inst.DIC())
        print("WAIC - model2 :", ic2_inst.WAIC())