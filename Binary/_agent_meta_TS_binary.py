from _util import *
import copy
import math

""" Adaption: 
episodic: update and sample every T
otherwise: every time
"""
from scipy.special import comb

def gamma_funcs(alpha,beta,positive,negative,tot,K):
    """ for a fixed j, calculate f(j) 
    read that paper again. really so hard?
    """
    return np.prod([comb(alpha[ii]+beta[ii], alpha[ii]) / comb(alpha[ii]+beta[ii]+tot[ii], alpha[ii]+positive[ii]) for ii in range(K)])
    
#     num=[math.gamma(alpha[a]+beta[a])*math.gamma(alpha[a]+positive[a])*math.gamma(beta[a]+negative[a]) for a in range(K)]
#     denom=[math.gamma(alpha[a])*math.gamma(beta[a])*math.gamma(alpha[a]+beta[a]+tot[a]) for a in range(K)]
#     result=[a/b for a,b in zip(num,denom)]
#     return np.prod(result)

class meta_TS_agent():
    """ in the experiment, we need to maintain N of them
    after each interaction, we will update the prior!
    when episodic, equalvalent
    """
    @autoargs()
    def __init__(self, T = None, order = None
                 , theta_prior_mean = None, theta_prior_cov = None
                 , u_prior_mean=None, ri_prior_phi_MC_fixed_theta=None
                 , phi_beta=None
                 , Xs = None # [N, K, p]
                 , N = 100, K = 2
                 , update_freq=1
                , N_candidates = 20
                , true_priors_4_debug = None
                ):
        """
        u_prior_mean: K_dim
        sigma : R
        sigma0 : prior
        sigmaq : meta prior
        """
        self.seed = 42
        self.K = K
        self.cnts = np.zeros((N, self.K))
        self.get_candi(phi_beta, theta_prior_mean, theta_prior_cov, u_prior_mean, ri_prior_phi_MC_fixed_theta
                      , N_candidates)
        self.Q = np.ones(N_candidates) / N_candidates #initial uniforma distribution of candidates, cand[0] is the alphabeta form u_prior_mean, prior_phi_beta

        self._init_posterior(N, self.K)
        
    #complete
    def get_candi(self,phi_beta, theta_prior_mean, theta_prior_cov, u_prior_mean, ri_prior_phi_MC_fixed_theta
                 , N_candidates):
        np.random.seed(self.seed)
        self.seed += 1
        cand_alpha1, cand_beta1 = beta_reparameterize(u_prior_mean, ri_prior_phi_MC_fixed_theta)
#         self.cand_alpha1, self.cand_beta1 = cand_alpha1, cand_beta1
        self.cand_ALPHA = [copy.deepcopy(cand_alpha1)]
        self.cand_BETA = [copy.deepcopy(cand_beta1)]
        #20 random theta from the prior
        cand_THETA = np.random.multivariate_normal(theta_prior_mean, theta_prior_cov, N_candidates - 1)
        #20 random X from the prior
        cand = choice(range(len(self.Xs)), N_candidates - 1)
        sample_PHI = self.Xs[cand, :, :]
        self.temp_means = []
        for i in range(N_candidates - 1):
            temp = sample_PHI[i].dot(cand_THETA[i])
            temp_mean = logistic(temp)
            temp_mean = np.squeeze(temp_mean)
            self.cand_ALPHA.append(beta_reparameterize(temp_mean, phi_beta)[0])
            self.cand_BETA.append(beta_reparameterize(temp_mean, phi_beta)[1])
#             self.cand_ALPHA.append([beta_reparameterize(mean, phi_beta)[0] for mean in temp_mean])
#             self.cand_BETA.append([beta_reparameterize(mean, phi_beta)[1] for mean in temp_mean])
#             print(self.cand_ALPHA[-1], temp_mean, "\n")
            self.temp_means.append(temp_mean)

    def _init_posterior(self, N, K):
        """ 
        """
        self.sample_prior_mean()
        #initialize the posterior of r
        self.posterior_alpha = [copy.deepcopy(self.r_prior_alpha) for i in range(N)]
        self.posterior_beta = [copy.deepcopy(self.r_prior_beta) for i in range(N)]
        
        self.posterior_positive_obs_ind = [zeros(K) for i in range(N)]
        self.posterior_negative_obs_ind = [zeros(K) for i in range(N)]
        
        #info to update meta posterior
        self.posterior_positive_obs_meta_post = zeros(K)
        self.posterior_negative_obs_meta_post = zeros(K)
        
        
    #complete
    def take_action(self, i, t, X = None):
        """ here is a 1-lag. so even with oracle, slightly worse"""
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.beta(self.posterior_alpha[i], self.posterior_beta[i], self.K)
        #Rs = [np.random.beta(a1, a2, 1) for a1, a2 in zip(self.posterior_alpha, self.posterior_beta)]
        A = np.argmax(Rs)
        return A
        
    def receive_reward(self, i, t, A, R, X = None, episode_finished = False):
        """
        only for the episodic condition now
        TODO: modified for the concurrent case

        """ 
         #Posterior updated each time based on the history of individual i
        self.posterior_positive_obs_ind[i][A] += R
        self.posterior_negative_obs_ind[i][A] += 1-R
#         print(self.r_prior_alpha[A].shape, self.posterior_positive_obs[i][A].shape)
#         if self.true_priors_4_debug is not None:
#         else:
        self.posterior_alpha[i] = self.r_prior_alpha + self.posterior_positive_obs_ind[i]
        self.posterior_beta[i] = self.r_prior_beta + self.posterior_negative_obs_ind[i]
        self.cnts[i,A] += 1
        
        self.posterior_positive_obs_meta_post[A] += R
        self.posterior_negative_obs_meta_post[A] += 1-R
        
        if self.order == "episodic":
            if episode_finished:
                self.update_meta_post()
                self.sample_prior_mean()
        elif self.order == "concurrent":
            if (i+1) % self.update_freq == 0:
                self.update_meta_post()
                self.sample_prior_mean()
           
    def update_meta_post(self):
        #update Q
        ALPHA = copy.deepcopy(self.cand_ALPHA)
        BETA = copy.deepcopy(self.cand_BETA)
        tot = self.posterior_positive_obs_meta_post + self.posterior_negative_obs_meta_post
        F = [gamma_funcs(ALPHA[j], BETA[j], self.posterior_positive_obs_meta_post, self.posterior_negative_obs_meta_post, 
                         tot, self.K) for j in range(len(ALPHA))]
        self.F = arr(F) / sum(F)
        self.Q = copy.deepcopy(self.Q) * F
        self.Q = arr(self.Q) / sum(self.Q)
        #self.Q = copy.deepcopy(self.Q)*F/np.sum(copy.deepcopy(self.Q)*F)
        #sample the new prior for next episode
        self.posterior_positive_obs_meta_post = zeros(self.K)
        self.posterior_negative_obs_meta_post = zeros(self.K)

    #complete
    def sample_prior_mean(self):
        np.random.seed(self.seed)
        self.seed += 1
#         print(range(len(self.cand_ALPHA)), len(self.Q))
        temp = np.random.choice(range(len(self.cand_ALPHA)), p=self.Q, size=1)
        temp = temp[0]
#         print(temp)
        self.r_prior_alpha = self.cand_ALPHA[temp]
        self.r_prior_beta = self.cand_BETA[temp]
        self.r_prior_alpha = np.squeeze(self.r_prior_alpha)
        self.r_prior_beta = np.squeeze(self.r_prior_beta)
    

        

            
