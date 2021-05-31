from _util import *

class LB_agent():
    """ "Lec 9: linear bandits and TS"
        1. prior_theta = prior_theta
        2. magnitute of the errors is the marginalized one 
    * already incremental
    """
    @autoargs()
    def __init__(self, sigma = 1
                 , prior_theta_u = None, prior_theta_cov = None
                 , N = 100, K = 2, p = 3):
        
        self.K = K
        self.cnts = np.zeros((N, self.K))
        
        self._init_posterior(self.K, p)
        self.seed = 42

    def _init_posterior(self, K, p):
        self.Cov = self.prior_theta_cov.copy()
        self.Cov_inv = inv(self.Cov)
        self.u = self.prior_theta_u.copy()
        
    def take_action(self, i, t, X):
        """
        X = [K, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        self.sampled_beta = np.random.multivariate_normal(self.u, self.Cov)
        self.sampled_Rs = X.dot(self.sampled_beta)
        self.A = np.argmax(self.sampled_Rs)
        
        return self.A
        
    def receive_reward(self, i, t, A, R, X):
        # update_data. update posteriors
        x = X[A]
        self.w_tilde = R - x.dot(self.sampled_beta)
        self.Cov_inv_last = self.Cov_inv.copy()
        self.Cov_inv += np.outer(x, x) / self.sigma ** 2
        self.Cov = inv(self.Cov_inv)
        
        self.u = self.Cov.dot(self.Cov_inv_last.dot(self.u) + x * (R + self.w_tilde / self.sigma ** 2))
        
        self.cnts[i, A] += 1
