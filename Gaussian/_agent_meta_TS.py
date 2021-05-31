from _util import *

""" Adaption: 
episodic: update and sample every T
otherwise: every time
"""

class meta_TS_agent():
    """ in the experiment, we need to maintain N of them
    after each interaction, we will update the prior!
    when episodic, equalvalent
    """
    @autoargs()
    def __init__(self, sigma = 1, order = None
                 , sigma1_square = None, theta = None 
                 , N = 100, K = 2, p = 10):
        """
        u_prior_mean: K_dim
        sigma : R
        sigma0 : prior
        sigmaq : meta prior
        """
        self.seed = 42
        self.K = K
        self.cnts = np.zeros((N, self.K))
        
        self.simga_q = 1 / p
        
        self.T_is = [0 for a in range(K)]
        self.sum_Y = [0 for a in range(K)]
        self.mean_ia_R = [zeros(N) for i in range(K)]
        self.sigma_i_sqrt = [np.repeat(self.sigma1_square, N) for i in range(K)]
        
        self.sigma0 = np.sqrt(theta[K:].dot(theta[K:]) + sigma1_square)
        #self.sigma0 = theta.dot(theta) + sigma1_square
        self.u_prior_cov = [self.sigma0 * identity(K) for i in range(N)]

        self.v = self.sigma ** -2
        self.v0 = self.sigma0 ** -2

        self._init_posterior(N, self.K, p = p)
        
        self.time_cost = [0, 0]

    def _init_posterior(self, N, K, p):
        """ 
        """
        
        self.meta_post = {"u" : zeros(K), "cov_diag" : self.simga_q * ones(K) }
        self.sample_prior_mean()
        self.posterior_u_num_wo_prior_mean = [zeros(K) for i in range(N)]
        self.posterior_u_num = [self.u_prior_mean / np.diag(self.u_prior_cov[i]) for i in range(N)]
        self.posterior_u_den = [1 / np.diag(self.u_prior_cov[i]) for i in range(N)]
        self.posterior_u = [self.posterior_u_num[i] / self.posterior_u_den[i] for i in range(N)] #zeros((N, K))
        self.posterior_cov_diag = [1 / self.posterior_u_den[i] for i in range(N)] #zeros((N, K))
        
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.multivariate_normal(self.posterior_u[i]
                                     , np.diag(self.posterior_cov_diag[i]))
        A = np.argmax(Rs)
        return A
        
    def receive_reward(self, i, t, A, R, X = None, episode_finished = False):
        # update_data. update posteriors
        self.posterior_u_num_wo_prior_mean[i][A] += (R / self.sigma ** 2)
        self.posterior_u_num[i][A] = self.posterior_u_num_wo_prior_mean[i][A] + self.u_prior_mean[A] / self.u_prior_cov[i][A, A]
        
        self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
        self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
        self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]
        
        self.cnts[i, A] += 1
        self.T_is[A] += 1
        self.sum_Y[A] +=  R

        if self.order == "episodic":
            if episode_finished:
                self.update_meta_post() # i, t, A, R
                self.sample_prior_mean()
        else:
            a = now()
            self.update_meta_post_concurrent(i, A, R) 
            self.time_cost[0] += (now() - a); a = now()
            self.sample_prior_mean()
            self.time_cost[1] += (now() - a); a = now()
        
        
    def sample_prior_mean(self):
        np.random.seed(self.seed)
        self.seed += 1
        self.u_prior_mean = np.random.multivariate_normal(self.meta_post['u'], np.diag(self.meta_post['cov_diag']))
    
    def update_meta_post(self): # , i, t, A, R
        """ 
        Posterior updated each time. 
        But not the mean in the posterior
        TODO: 
            2. do not loop
            3. the overall schedule
        """
        for a in range(self.K):
            v_hat = 1 / self.meta_post['cov_diag'][a]
            c1 = self.v0 + self.T_is[a] * self.v
            c2 = v_hat + self.v0 - 1 / c1 * self.v0 ** 2
            self.meta_post['u'][a] = 1 / c2 * (1 / c1 * self.v0 * self.v * self.sum_Y[a] + v_hat * self.meta_post['u'][a])
            self.meta_post['cov_diag'][a] = 1 / c2
            self.T_is[a] = 0
            self.sum_Y[a] =  0

    def update_meta_post_concurrent(self, i, A, R):
        # maintain sigma_i
        d = 1 / self.simga_q
        self.mean_ia_R[A][i] = (self.mean_ia_R[A][i] * (self.cnts[i, A] - 1) + R) / self.cnts[i, A]
        self.sigma_i_sqrt[A][i] = self.sigma1_square + self.sigma ** 2 / self.cnts[i, A]
        
        avail_tasks = np.where(self.cnts[:, A])
        
        num = np.sum(self.mean_ia_R[A][avail_tasks] / self.sigma_i_sqrt[A][avail_tasks])
        den = np.sum(1 / self.sigma_i_sqrt[A][avail_tasks]) + d
        self.meta_post['u'][A] = num / den
        self.meta_post['cov_diag'][A] = 1 / den
        