from _util import *

class Environment():
    @autoargs()
    def __init__(self, N, T, K, p_raw, p, sigma, Sigma_delta
                 , u_theta, Sigma_theta
                , seed = 42, with_intercept = True
                 , X_mu = None, X_Sigma = None
                 , misspecification = None
                ):
        self.setting = locals()
        self.setting['self'] = None
        self.seed = seed
        np.random.seed(self.seed)
        
        self.theta = np.random.multivariate_normal(u_theta, Sigma_theta, 1)[0]

#         self.simu_X(N)
        self.get_Phi(N, p, K)
        self.get_r(K, N, Sigma_delta)
        self.optimal_arms = np.argmax(self.r, 1) # N-dim
        self.errors = randn(N, T, K) * sigma
#         self.simu_errors(N, T, K, sigma)
                
         
    def get_Phi(self, N, p, K):
        """ consider the simple case now 
        [N, K, p]
        TODO
        """
        np.random.seed(self.seed)
        if self.with_intercept:
            self.intercept_Phi = np.repeat(identity(K)[np.newaxis, :, :], N, axis = 0)
            self.random_part_Phi = np.random.multivariate_normal(self.X_mu[K:], self.X_Sigma[K:][:, K:], (N, K))
            self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 2)
        else:
            self.Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (N, K))

#         self.X_mu = np.zeros(p - K)
#         self.X_Sigma = identity(p - K)
#         # [N, K, p]
#         self.intercept_Phi = np.repeat(identity(K)[np.newaxis, :, :], N, axis = 0)
#         self.random_part_Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (N, K))
#         self.Phi = np.concatenate([self.intercept_Phi, self.random_part_Phi], axis = 2)
        
#         self.X_mu = np.zeros(p)
#         self.X_Sigma = identity(p)
#         self.Phi = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (N, K))
    
    def get_r(self, K, N, Sigma_delta):
        """ 
        TODO : misspecifications can be added here. nonlinear as the true model to show the robustness w.r.t. LMM 
        """
        np.random.seed(self.seed)
        # [N, K]
        self.deltas = np.random.multivariate_normal(np.zeros(K), Sigma_delta, N)
        self.r_mean = [Phi_i.dot(self.theta) for Phi_i in self.Phi]
        if self.misspecification is not None:
            w_linear, w_non_linear = self.misspecification[1]
            if self.misspecification[0] == "sin":
                r_max = max([np.max(a) for a in self.r_mean])
                self.r_mean = [self.approximate_linear_with_sin(x, r_max) * w_non_linear + x * w_linear for x in self.r_mean]
            elif self.misspecification[0] == "cos":
                r_max = max([np.max(a) for a in self.r_mean])
                self.r_mean = [self.approximate_linear_with_cos(x, r_max) * w_non_linear + x * w_linear for x in self.r_mean]
        self.r = [r_mean + delta for r_mean, delta in zip(self.r_mean, self.deltas)]
        self.r = np.vstack(self.r)
    
    def get_reward(self, i, t, a):
        return self.r[i, a] + self.errors[i, t, a]
    
    def get_optimal_reward(self, i, t):
        a = self.optimal_arms[i]
        return (self.r[i] + self.errors[i, t])[a]
#     def simu_errors(self, N, T, K, sigma):
#         self.errors = randn(N, T, K) * sigma

#     def approximate_linear_with_sin(self, x, max_x):
#         """
#         sin is close to x between [-pi / 2, pi / 2]
#         1. x back to this range
#         2. get the value of sin(x)
#         3. transform back
#         """
#         factor = 1 / max_x * np.pi / 2
#         return np.sin(x * factor) / factor
    def approximate_linear_with_cos(self, x, max_x):
        """
        sin is close to x between [-pi / 2, pi / 2]
        1. x back to this range
        2. get the value of sin(x)
        3. transform back
        """
        factor = 1 / max_x * np.pi / 2
        return np.cos(x * factor) / factor
