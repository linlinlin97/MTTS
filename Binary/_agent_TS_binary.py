from _util import *
import copy

def beta_reparameterize(pi,phi_beta):
    return pi / phi_beta, (1 - pi) / phi_beta

class TS_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, u_prior_mean, prior_phi_beta):
        ### R ~ bernoulli(ri)
        ### ri~beta(mean, phi)
                
        self.K = len(u_prior_mean)
        self.u_prior_alpha,self.u_prior_beta = beta_reparameterize(u_prior_mean, prior_phi_beta)
        
        self.cnts = zeros(self.K)
        self._init_posterior(self.K)
        self.seed = 42

    def _init_posterior(self, K):
        self.posterior_alpha = copy.deepcopy(self.u_prior_alpha)
        self.posterior_beta = copy.deepcopy(self.u_prior_beta)
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.beta(self.posterior_alpha, self.posterior_beta, self.K)
        A = np.argmax(Rs)
        return A
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_alpha[A] += R
        self.posterior_beta[A] += 1-R
        self.cnts[A] += 1
        
class N_TS_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, u_prior_mean, prior_phi_beta, N = 100):
        ### R ~ bernoulli(ri)
        ### ri~beta(mean, phi)
                
        self.K = len(u_prior_mean)
        self.N = N
        self.u_prior_alpha, self.u_prior_beta = beta_reparameterize(u_prior_mean, prior_phi_beta)
        self.cnts = np.zeros((N, self.K))
        self._init_posterior(N)
        self.seed = 42
        
    def _init_posterior(self, N):
        self.posterior_alpha = [copy.deepcopy(self.u_prior_alpha) for ind in range(self.N)] #zeros((N, K))
        self.posterior_beta = [copy.deepcopy(self.u_prior_beta) for ind in range(self.N)] #zeros((N, K))
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.beta(self.posterior_alpha[i], self.posterior_beta[i], self.K)
        A = np.argmax(Rs)
        return A
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_alpha[i][A] += R
        self.posterior_beta[i][A] += 1-R
        self.cnts[i, A] += 1

        
class meta_oracle_agent():
    """ in the experiment, we need to maintain N of them
    """
    @autoargs()
    def __init__(self, u_prior_alpha = None, u_prior_beta = None):
        ### R ~ N(mu, sigma)
        ### sigma as known
        ### prior over mu
        
        self.K = K = len(u_prior_alpha[0])
        self.N = N = len(u_prior_alpha)
        self.cnts = np.zeros((N, self.K))
        self._init_posterior(N, self.K)
        self.seed = 42
        
    def _init_posterior(self, N, K):
        self.posterior_alpha = copy.deepcopy(self.u_prior_alpha)
        self.posterior_beta = copy.deepcopy(self.u_prior_beta)
        
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
        Rs = np.random.beta(self.posterior_alpha[i], self.posterior_beta[i], self.K)
        A = np.argmax(Rs)
        return A
        
    def receive_reward(self, i, t, A, R, X = None):
        # update_data. update posteriors
        self.posterior_alpha[i][A] += R
        self.posterior_beta[i][A] += 1-R
        self.cnts[i, A] += 1
