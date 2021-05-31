from _util import *
# !conda install -c conda-forge pymc3==3.11.2 --force-reinstall -y

# !gcc --version
# !conda install -c conda-forge pymc3
# !sudo yum install gcc-c++ -y
# !conda update python --y
#from pymc3.distributions import Interpolated
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


########################################################################
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
logger1 = logging.getLogger('theano')
logger1.setLevel(logging.ERROR)

_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.ERROR)
_logger = logging.getLogger("INFO (theano.gof.compilelock)")
_logger.setLevel(logging.ERROR)
########################################################################


# incremental: https://stackoverflow.com/questions/53211277/updating-model-on-pymc3-with-new-observed-data


class MTB_agent():
    @autoargs()
    def __init__(self, start = "cold", phi_beta = None, T = None, order = None # start==cold start or warm start
                 , theta_prior_mean = None, theta_prior_cov = None
                 , Xs = None # [N, K, p]
                 , update_freq = 100
                , u_prior_alpha = None, u_prior_beta = None
                 , true_theta_4_debug = None, true_r_4_debug = None
                 , exp_seed = None
                ):
        self.seed = 42 #random seed
        self.K = K = Xs.shape[1]
        self.N = N = len(Xs) #[N,K,P]
        self.p = p = len(theta_prior_mean)
        self.theta_prior_cov = theta_prior_cov
        self.Phi = Xs
        self.Phi_pos = np.vstack(Xs)#[K*N,p]
        self._init_prior_posterior(N,K)
        self.traces = []
        self.cnts = np.zeros((N, self.K)) #record number of pulls for each individual each action
#         self.used_first_prior = {}
#         self.debug = {}
        self.recorder = {"sampled_thetas" : []
                        , "sampled_priors" : []}      
        
        
    def _init_prior_posterior(self, N,K):
        self.theta = np.random.multivariate_normal(self.theta_prior_mean, self.theta_prior_cov)
        self.theta_2_alpha_beta() #get r prior alpha and beta
        
        if self.start == "oracle_4_debug":
            ### get from oracle-TS
            self.r_prior_alpha = self.u_prior_alpha
            self.r_prior_beta = self.u_prior_beta    
        
        #initialize the posterior of r
        self.posterior_alpha = copy.deepcopy(self.r_prior_alpha)
        self.posterior_beta = copy.deepcopy(self.r_prior_beta)
        
        self.posterior_alpha_wo_prior_mean = [zeros(K) for i in range(N)]
        self.posterior_beta_wo_prior_mean = [zeros(K) for i in range(N)]
        
        self.idx = []
        self.n_trails = zeros(N * K)
        self.n_success = zeros(N * K)
    

    ########################################################################################################
    ########################################################################################################
    #take action after encountering task i with X at time t (complete)
    def take_action(self, i, t, X = None):
        np.random.seed(self.seed)
        self.seed += 1
#         if self.start == "oracle_4_debug":
#             A = np.random.choice(self.K)
#         else:        
        Rs = np.random.beta(self.posterior_alpha[i], self.posterior_beta[i], self.K)
        A = np.argmax(Rs)
#         if i == 2:
#             self.debug[t] = [Rs, self.posterior_alpha[i], self.posterior_beta[i]]

        return A
    
    def receive_reward(self, i, t, A, R, X = None, episode_finished = False):
        """update_data, and posteriors
        record ia with data, the number of trails and the number of success. 
        """
#         if t == 0:
#             self.used_first_prior[i] = [self.r_prior_alpha[i], self.r_prior_beta[i]]
                
        temp_index = i * self.K + A #ID*K+Action
        if temp_index not in self.idx:
            self.idx.append(int(temp_index))
        self.n_trails[temp_index] += 1
        self.n_success[temp_index] += R
        
        # Posterior updated each time based on the history of individual i
        self.posterior_alpha_wo_prior_mean[i][A] += R
        self.posterior_beta_wo_prior_mean[i][A] += (1 - R)
        ### some bugs here, at first.
        self.posterior_alpha[i] = self.r_prior_alpha[i] + self.posterior_alpha_wo_prior_mean[i]
        self.posterior_beta[i] = self.r_prior_beta[i] + self.posterior_beta_wo_prior_mean[i]
        self.cnts[i, A] += 1
        

        if self.start != "oracle_4_debug" and ((self.order == "episodic" and episode_finished and (i % self.update_freq == 0 or i < 10)) or (self.order == "concurrent" and (i + 1) % self.N == 0 and (t % self.update_freq == 0 or t < 10))):
#             if self.exp_seed % 16 == 0:
#             print(self.order, self.exp_seed, i, t)
            self.sample_theta(i, t)
            self.theta_2_alpha_beta()

    ######################################## get the posterior of theta by all history information ####################################        
    def sample_theta(self, i, t):
        """ randomly select a theta from posterior based on all history information, using Pymc3
        """
        K, p, N = self.K, self.p, self.N
        """ MCMC_scheduler 
        TODO: true incremental: less data points. 
        #np.arange(1000, 200, -20)
        The only difference now: trace
        """
        n_init = 2000
        if self.order == "episodic":
            n_sample = n_init - int(i * 8)
        elif self.order == "concurrent":
            n_sample = n_init - int(t * 8)
        n_tune = 100
        chains = 1
        self.temp_idx = np.array([int(x) for x in self.idx])
        x = self.Phi_pos[self.temp_idx]
        R = self.n_success[self.temp_idx].astype('int32')
        n_trails = self.n_trails[self.temp_idx].astype('int32')
        
#         np.random.seed(42)
#         x = self.Phi_pos
#         n_trails = np.repeat(50, len(x))
#         R = np.random.binomial(n_trails, self.true_r_4_debug)
        with pm.Model() as beta_bernoulli:
            theta_temp = pm.MvNormal('theta', mu=self.theta_prior_mean, cov=self.theta_prior_cov,shape=p)
            alpha_temp = pm.math.dot(x, theta_temp)
            mean_beta = logistic(alpha_temp)
            alpha_Beta, beta_Beta = beta_reparameterize(mean_beta, self.phi_beta)
#             obs = pm.Beta('obs', alpha = alpha_Beta, beta = beta_Beta, observed= self.true_r_4_debug[self.temp_idx])

            obs = pm.BetaBinomial('obs', alpha_Beta, beta_Beta, n_trails, observed = R)
            if len(self.traces)==0:
                last_trace = None
            else:
                last_trace = self.trace
            trace = pm.sample(n_sample, tune = n_tune, chains = chains
                              , cores = 1, progressbar = 0
#                                   , nuts_kwargs={'target_accept': 0.95}
                             , target_accept = 0.85 # default = 0.8
                              , trace = last_trace # make worse? or what?           
                              # max_treedepth = 10 # default = 10
                             )
        self.trace = trace #update trace
        self.traces.append(trace)
        self.theta = trace["theta"][-1] #random.choice(trace["theta"])
        self.theta_mean = np.mean(trace["theta"], 0)
        

        self.recorder["sampled_thetas"].append(self.theta)
        if (self.exp_seed % 5 == 0) and (i % 10 == 0):
            self.mse_mean_theta = arr([self.RMSE_theta(np.mean(trace1["theta"], 0)) for trace1 in self.traces])
            self.mse_sampled_theta = arr([self.RMSE_theta(trace1["theta"][-1]) for trace1 in self.traces])
            self.std_theta = arr([np.mean(np.std(trace1["theta"], 0)) for trace1 in self.traces])
            pd.set_option("display.precision", 3)

            result = np.array([self.mse_mean_theta, self.mse_sampled_theta, self.std_theta])
            s = DF(result, index=["mean", "sampled", "std"])
            display(s)

#     def sample_theta_quick(self):
#         np.random.seed(self.seed)
#         self.seed += 1
#         try:
#             idx = np.random.choice(len(self.trace["theta"]))
#             self.theta = self.trace["theta"][idx]
#         except:
#             self.theta = np.random.multivariate_normal(self.theta_prior_mean, self.theta_prior_cov)
        
        
    def theta_2_alpha_beta(self):
        """ get the prior for each task"""
        alpha_temp = [Phi_i.dot(self.theta) for Phi_i in self.Phi]
        self.sample_beta_prior_mean = logistic(alpha_temp)
        #list of N elements, each with an (3,) array
        self.r_prior_alpha = [beta_reparameterize(mean, self.phi_beta)[0] for mean in self.sample_beta_prior_mean]
        self.r_prior_beta = [beta_reparameterize(mean, self.phi_beta)[1] for mean in self.sample_beta_prior_mean]
        
        
    def RMSE_theta(self, v):
        return np.sqrt(np.mean((v - self.true_theta_4_debug) **2))
    
# def prior_from_posterior(param, samples):
#     smin, smax = np.min(samples), np.max(samples)
#     width = smax - smin
#     x = np.linspace(smin, smax, 100)
#     y = stats.gaussian_kde(samples)(x)
    
#     # what was never sampled should have a small probability but not 0,
#     # so we'll extend the domain and use linear approximation of density on it
#     x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
#     y = np.concatenate([[0], y, [0]])
#     return Interpolated(param, x, y)
