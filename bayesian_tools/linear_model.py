from random import seed, normalvariate

import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    from MCMC_Core import MCMC_base, MCMC_Gibbs
    from info_criteria import InfomationCriteria
    from sampler_gamma import Sampler_univariate_InvChisq, Sampler_univariate_InvGamma
else:
    from bayesian_tools.info_criteria import InfomationCriteria
    from bayesian_tools.MCMC_Core import MCMC_base, MCMC_Gibbs
    from bayesian_tools.sampler_gamma import Sampler_univariate_InvChisq, Sampler_univariate_InvGamma

class LM_noninfo_prior(MCMC_base):
    #prior p(mu, sigma^2) ~ sigma^(-2)
    def __init__(self, response_vec, design_matrix, rnd_seed) -> None:
        self.x = design_matrix
        self.y = response_vec
            
        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1] #k+1

        self.MC_sample = []

        seed(rnd_seed)
        self.inv_chisq_sampler = Sampler_univariate_InvChisq()
        self.np_rng = np.random.default_rng()

        self.df = self.n - self.dim_beta
        self.xtx_inv = np.linalg.inv(np.transpose(self.x) @ self.x)
        self.beta_hat = self.xtx_inv @ np.transpose(self.x) @ self.y
        self.residual = self.y - (self.x @ self.beta_hat)
        self.s2_without_div_df = np.dot(self.residual, self.residual)

    def print_freqentist_result(self):
        print("beta_hat:", self.beta_hat)
        print("s2_without_div_df:", self.s2_without_div_df)
        print("s2_with_div_df:", self.s2_without_div_df/self.df)
        beta_hat_cov_mat = self.xtx_inv * (self.s2_without_div_df/self.df)
        beta_hat_var_list = [beta_hat_cov_mat[i,i] for i in range(beta_hat_cov_mat.shape[0])]
        print("beta_hat_var:", beta_hat_var_list)

    def sampler(self, **kwarg):
        inv_chisq_sample = self.inv_chisq_sampler.sampler_iter(1, self.df)[0]
        new_sigma = self.s2_without_div_df * inv_chisq_sample
        new_beta = self.np_rng.multivariate_normal(self.beta_hat, self.xtx_inv * new_sigma)
        self.MC_sample.append([new_sigma]+[x for x in new_beta])
    

class LM_random_eff_fixed_slope_noninfo_prior(MCMC_Gibbs):
    #prior p(mu, sigma^2) ~ 1*inv_gamma(self.hyper_tau2_0_shape, self.hyper_tau2_0_rate)
    #when 0, it get to be p(mu,sigma^2)~ sigma^(-2)
    #however, improper prior may cause a stuck in posterior
    def __init__(self, response_vec, design_matrix, rand_eff_group_col_indicator_list, initial, rnd_seed) -> None:
        # rand_eff_group_indicator_list: 1 if the variable is in the group / 0 if not
        #now, this class support only 'one' random effect group.
        self.rand_eff_group = rand_eff_group_col_indicator_list
        self.x = design_matrix
        self.y = response_vec
        self.hyper_tau2_1 = 10000
        self.hyper_mu1 = 0
        self.hyper_tau2_0_shape = 0.1 # tune here if needed
        self.hyper_tau2_0_rate = 0.1 # tune here if needed

        self.n = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1]

        self.MC_sample = [initial]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]

        seed(rnd_seed)
        self.inv_gamma_sampler = Sampler_univariate_InvGamma()
        self.np_rng = np.random.default_rng()

        self.xtx = np.transpose(self.x) @ self.x
        self.xtx_inv = np.linalg.inv(self.xtx)
        self.xty = np.transpose(self.x) @ self.y

    def _full_conditional_sampler_beta(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        D_inv_list = []
        m_list = []
        for ind in self.rand_eff_group:
            if ind==0:
                D_inv_list.append(1/self.hyper_tau2_1)
                m_list.append(self.hyper_mu1)
            elif ind==1:
                D_inv_list.append(1/new_sample[3])
                m_list.append(new_sample[2])
            else:
                raise ValueError("check your random effect group indicator list.")
        D_inv = np.diag(D_inv_list)
        m = np.array(m_list)

        beta_precision = self.xtx / new_sample[1] + D_inv
        beta_variance = np.linalg.inv(beta_precision)
        beta_mean = beta_variance @ (self.xty / new_sample[1] + D_inv @ m)
        new_beta = self.np_rng.multivariate_normal(beta_mean, beta_variance)
        new_sample[0] = new_beta
        return new_sample

    def _full_conditional_sampler_sigma2(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        sigma2_shape = self.n/2
        resid = self.y - (self.x@new_sample[0])
        sigma2_rate = np.dot(resid, resid)/2
        new_sigma2 = self.inv_gamma_sampler.sampler(sigma2_shape, sigma2_rate)
        new_sample[1] = new_sigma2
        return new_sample

    def _full_conditional_sampler_mu0(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        mu0_sum = 0
        num_group_member = 0
        for i, ind in enumerate(self.rand_eff_group):
            if ind==1:
                num_group_member += 1
                mu0_sum += new_sample[0][i]
        mu0_mean = mu0_sum/num_group_member
        mu0_var = new_sample[3]/num_group_member
        new_mu0 = normalvariate(mu0_mean, mu0_var**0.5)
        new_sample[2] = new_mu0
        return new_sample

    def _full_conditional_sampler_tau2_0(self, last_param):
        new_sample = [np.array([beta_i for beta_i in last_param[0]])] + [last_param[i] for i in range(1,4)]
        #  0       1       2    3
        # [[beta], sigma2, mu0, tau2_0]
        tau2_rate = 0
        num_group_member = 0
        for i, ind in enumerate(self.rand_eff_group):
            if ind==1:
                num_group_member += 1
                tau2_rate += ((new_sample[0][i]-new_sample[2])**2)/2

        tau2_shape = num_group_member/2
        new_tau2 = self.inv_gamma_sampler.sampler(tau2_shape+self.hyper_tau2_0_shape, tau2_rate+self.hyper_tau2_0_rate)
        new_sample[3] = new_tau2
        return new_sample

    def sampler(self, **kwarg):
        last_sample = self.MC_sample[-1]
        new_sample = self._full_conditional_sampler_beta(last_sample)
        new_sample = self._full_conditional_sampler_sigma2(new_sample)
        new_sample = self._full_conditional_sampler_mu0(new_sample)
        new_sample = self._full_conditional_sampler_tau2_0(new_sample)
        self.MC_sample.append(new_sample)
    


class InfomationCriteria_for_LM(InfomationCriteria):
    def __init__(self, response_vec, design_matrix, beta_samples, sigma2_samples):
        self.beta_samples = beta_samples
        self.sigma2_samples = sigma2_samples
        self.y = response_vec
        self.x = design_matrix
        self.data = [(y,x) for y, x in zip(self.y, self.x)]
        self.MC_sample = np.array([(s,b) for s, b in zip(self.sigma2_samples, self.beta_samples)], dtype=object)

    def _dic_log_likelihood_given_full_data(self, param_vec):
        sigma2 = param_vec[0]
        beta = param_vec[1]
        n = len(self.x)
        residual = self.y-(self.x@beta)
        exponent = np.dot(residual, residual) / (-2*sigma2)
        return (-n/2)*np.log(sigma2) + exponent
    
    def _waic_regular_likelihood_given_one_data_pt(self, param_vec, data_point_vec):
        sigma2 = param_vec[0]
        beta = param_vec[1]
        y, x = data_point_vec
        residual = y-(x@beta)
        exponent = np.dot(residual, residual) / (-2*sigma2)
        return sigma2**(-1/2) * np.exp(exponent)

class Regression_Model_Checker:
    def __init__(self, response_vec, design_mat, beta_samples, sigma2_samples):
        self.y = response_vec
        self.x = design_mat
        self.beta_samples = beta_samples
        self.sigma2_samples = sigma2_samples

        self.mean_beta = np.mean(self.beta_samples, axis=0)
        self.mean_sigma2 = np.mean(self.sigma2_samples)
    
        self.fitted = self.x @ self.mean_beta
        self.residuals = self.y - self.fitted
        self.standardized_residuals = self.residuals/(self.mean_sigma2**0.5)

    def show_residual_plot(self, show=True):
        x_axis = self.fitted
        y_axis = self.residuals
        plt.plot(x_axis, y_axis, 'bo')
        plt.xlabel("y-fitted")
        plt.ylabel("standardized residual")
        plt.title("residual plot")
        plt.axhline(0)
        plt.axhline(1.96, linestyle='dashed')
        plt.axhline(-1.96, linestyle='dashed')
        if show:
            plt.show()

    def show_residual_normalProbplot(self, show=True):
        from scipy.stats import probplot
        probplot(self.residuals, plot=plt)
        plt.xlabel("theoretical quantiles")
        plt.ylabel("observed values")
        plt.title("normal probability plot")

        if show:
            plt.show()

    def show_posterior_predictive_at_new_point(self, design_row, reference_response_val=None, show=True, color=None, x_lab=None, x_lim=None):
        predicted = []
        for beta, sigma2 in zip(self.beta_samples, self.sigma2_samples):
            new_y = (design_row @ beta) + normalvariate(0, sigma2**0.5)
            predicted.append(new_y)
        
        if color is None:
            plt.hist(predicted, bins=50, density=True, histtype="step")
            if reference_response_val is not None:
                plt.axvline(reference_response_val)
        else:
            desig_color = "C"+str(color%10)
            plt.hist(predicted, bins=50, density=True, histtype="step", color=desig_color)
            if reference_response_val is not None:
                plt.axvline(reference_response_val, color=desig_color)
        if x_lab:
            plt.xlabel(x_lab)
        else:
            plt.xlabel("predicted at:"+str(design_row))
        if x_lim:
            plt.xlim(x_lim)
        if show:
            plt.show()        

    def show_posterior_predictive_at_given_data_point(self, data_idx, show=True, x_lab=None):
        design_row = self.x[data_idx,:]
        ref_y = self.y[data_idx]
        self.show_posterior_predictive_at_new_point(design_row, ref_y, show, color=data_idx, x_lab=x_lab)






if __name__=="__main__":
    # test_x = np.array([[1,x*0.5] for x in range(30)])
    # from random import normalvariate
    # test_y = test_x[:,0]*2 + test_x[:,1]*1.3 + np.array([normalvariate(0,0.1) for _ in test_x])
    # print(test_y)

    # lm_inst = LM_noninfo_prior(test_y, test_x, 20220519)
    # lm_inst.generate_samples(10000)
    # lm_inst.print_freqentist_result()
    
    # from MCMC_Core import MCMC_Diag
    # diag_inst = MCMC_Diag()
    # diag_inst.set_mc_sample_from_MCMC_instance(lm_inst)
    # diag_inst.set_variable_names(["sigma2", "beta0", "beta1"])
    # diag_inst.print_summaries(round=8)
    # diag_inst.show_hist((1,3))
    # diag_inst.show_scatterplot(1,2)
    

    test2_x = np.array([
        [1,0,0,0,1,1],
        [1,0,0,0,2,1],
        [1,0,0,0,3,2],
        [1,0,0,0,4,2],
        [1,0,0,0,5,3],
        [1,0,0,0,6,3],
        [1,0,0,0,7,3],

        [0,1,0,0,1,1],
        [0,1,0,0,2,1],
        [0,1,0,0,3,2],
        [0,1,0,0,4,2],
        [0,1,0,0,5,3],
        [0,1,0,0,6,3],
        [0,1,0,0,7,3],

        [0,0,1,0,-1,1],
        [0,0,1,0,-2,1],
        [0,0,1,0,-3,2],
        [0,0,1,0,-4,2],
        [0,0,1,0,-5,3],
        [0,0,1,0,-6,3],
        [0,0,1,0,-7,3],

        [0,0,0,1,-1,1],
        [0,0,0,1,-2,1],
        [0,0,0,1,-3,2],
        [0,0,0,1,-4,2],
        [0,0,0,1,-5,3],
        [0,0,0,1,-6,3],
        [0,0,0,1,-7,3],
        ])
    test2_y = test2_x[:,0]*(-1) + test2_x[:,1]*3 + test2_x[:,2]*(-2) + test2_x[:,3]*1 + test2_x[:,4]*1 + test2_x[:,5]*0.5 + np.array([normalvariate(0, 0.4) for _ in test2_x])
    print(test2_y)
    test2_indicator = [1,1,1,1,0,0]
    #  0       1       2    3
    # [[beta], sigma2, mu0, tau2_0]
    test2_initial = [np.array([0,0,0,0,0,0]),1, 0, 1]
    lm_inst2 = LM_random_eff_fixed_slope_noninfo_prior(test2_y, test2_x, test2_indicator, test2_initial, 20220519)
    lm_inst2.generate_samples(10000, print_iter_cycle=5000)
    
    
    from MCMC_Core import MCMC_Diag
    diag_inst21 = MCMC_Diag()
    betas2 = [x[0] for x in lm_inst2.MC_sample]
    diag_inst21.set_mc_samples_from_list(betas2)
    diag_inst21.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5"])
    diag_inst21.print_summaries(round=8)
    # diag_inst21.show_hist((1,6))
    # diag_inst21.show_traceplot((1,6))

    diag_inst22 = MCMC_Diag()
    others2 = [x[1:4] for x in lm_inst2.MC_sample]
    diag_inst22.set_mc_samples_from_list(others2)
    diag_inst22.set_variable_names(["sigma2", "mu0", "tau2_0"])
    diag_inst22.print_summaries(round=8)
    # diag_inst22.show_hist((1,3))
    # diag_inst22.show_traceplot((1,3))


    checker_inst2 = Regression_Model_Checker(test2_y, test2_x, betas2, diag_inst22.get_specific_dim_samples(0))
    checker_inst2.show_residual_plot()
    checker_inst2.show_residual_normalProbplot()
    checker_inst2.show_posterior_predictive_at_new_point([0,0,0,1,1,1], reference_response_val=2.5)
    checker_inst2.show_posterior_predictive_at_given_data_point(0)
