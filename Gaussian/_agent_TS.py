from _util import *
# pip install thompson-sampling
class TS_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, sigma = 1, u_prior_mean = None, u_prior_cov = None):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu
                
        self.K = len(u_prior_mean)
        self.cnts = zeros(self.K)
        self._init_posterior(self.K)
        self.seed = 42

    def _init_posterior(self, K):
        self.posterior_u_num = self.u_prior_mean / np.diag(self.u_prior_cov)
        self.posterior_u_den = 1 / np.diag(self.u_prior_cov)
        self.posterior_u = zeros(K)
        self.posterior_cov_diag = zeros(K)
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.multivariate_normal(self.posterior_u
                                     , np.diag(self.posterior_cov_diag))
        return np.argmax(Rs)
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_u_num[A] += (R / self.sigma ** 2)
        self.posterior_u_den[A] += (1 / self.sigma ** 2)
        self.posterior_u[A] = self.posterior_u_num[A] / self.posterior_u_den[A]
        self.posterior_cov_diag[A] = 1 / self.posterior_u_den[A]
        self.cnts[A] += 1
"""
Q: anything wrong with the posterior updating?
"""


class N_TS_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, sigma = 1, u_prior_mean = None, u_prior_cov = None, N = 100):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu
                
        self.K = len(u_prior_mean)
        self.cnts = np.zeros((N, self.K))
        self._init_posterior(N, self.K)
        self.seed = 42
        
    def _init_posterior(self, N, K):
        self.posterior_u_num = [self.u_prior_mean / np.diag(self.u_prior_cov) for _ in range(N)]
        self.posterior_u_den = [1 / np.diag(self.u_prior_cov) for _ in range(N)]
        self.posterior_u = [self.posterior_u_num[i] / self.posterior_u_den[i] for i in range(N)] #zeros((N, K))
        self.posterior_cov_diag = [1 / self.posterior_u_den[i] for i in range(N)] #zeros((N, K))
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.multivariate_normal(self.posterior_u[i]
                                     , np.diag(self.posterior_cov_diag[i]))
        return np.argmax(Rs)
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_u_num[i][A] += (R / self.sigma ** 2)
        self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
        self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
        self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]
        self.cnts[i, A] += 1

        
class meta_oracle_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, sigma = 1, u_prior_mean = None, u_prior_cov = None):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu
        
        self.K = K = len(u_prior_cov[0])
        self.N = N = len(u_prior_cov)
        self.cnts = np.zeros((N, self.K))
        self._init_posterior(N, self.K)
        self.seed = 42
        
    def _init_posterior(self, N, K):
        self.posterior_u_num = [self.u_prior_mean[i] / np.diag(self.u_prior_cov[i]) for i in range(N)]
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
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_u_num[i][A] += (R / self.sigma ** 2)
        self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
        self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
        self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]
        self.cnts[i, A] += 1
