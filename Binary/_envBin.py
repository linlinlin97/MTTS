from _util import *

class Environment():
    @autoargs()
    def __init__(self, N, T, K, p_raw, p, phi_beta
                 , u_theta, Sigma_theta
                , seed = 42, with_intercept = True
                 , X_mu = None, X_Sigma = None
                 , misspecification = None
                ):
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        
        self.X_mu = X_mu
        self.X_Sigma = X_Sigma
        
        self.theta = np.random.multivariate_normal(u_theta, Sigma_theta, 1)[0]

        #self.simu_X(N)
        self.get_Phi(N, K)
        self.get_r(K, N, phi_beta)
        self.optimal_arms = np.argmax(self.r, 1) # N-dim
        
        self.generate_all_random_reward(N, T, K)
        
    def get_Phi(self, N, K):
        """ consider the simple case now 
        [N, K, p]
        """
        np.random.seed(self.seed)
        if self.with_intercept:
            self.intercept_Phi = np.repeat(identity(K)[np.newaxis, :, :], N, axis = 0)
            self.random_part_Phi = np.random.multivariate_normal(self.X_mu[K:], self.X_Sigma[K:][:, K:], (N, K))
            self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 2)
        else:
            self.Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (N, K))
        
    
    def get_r(self, K, N, phi_beta):
        """ simulate r_i based on BBLM
        self.r = [N,K]
        """
        np.random.seed(self.seed)
        alpha_temp = [Phi_i.dot(self.theta) for Phi_i in self.Phi]
        self.r_mean = np.exp(alpha_temp)/(np.exp(alpha_temp)+1)
        self.alpha_Beta = [beta_reparameterize(mean, phi_beta)[0] for mean in self.r_mean]
        self.beta_Beta = [beta_reparameterize(mean, phi_beta)[1] for mean in self.r_mean]
        self.r = [np.random.beta(alpha_i, beta_i, K) for alpha_i, beta_i in zip(self.alpha_Beta, self.beta_Beta)]
        self.r = np.vstack(self.r)

    def generate_all_random_reward(self, N, T, K):
        """ generate a matrix of dimension (T, N, K)
        for each (i, a), we sample T rewards
        """
        np.random.seed(self.seed)
        self.Rs = np.random.binomial(1, self.r, size = (T, N, K))
    
    #get the obaseved R_it if i t a 
    def get_reward(self, i, t, a):
        return self.Rs[t, i, a]
    
    #get the observed reward if pulling the optimal arm
    def get_optimal_reward(self, i, t):
        a = self.optimal_arms[i]
        return self.Rs[t, i, a]

