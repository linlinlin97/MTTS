from _util import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import Gaussian._env as _env
reload(_env)

import Gaussian._agent_MTB as _agent_MTB
import Gaussian._agent_LB as _agent_LB
import Gaussian._agent_TS as _agent_TS
import Gaussian._agent_meta_TS as _agent_meta_TS

reload(_agent_MTB)
reload(_agent_LB)
reload(_agent_TS)

reload(_agent_meta_TS)


import Binary._envBin as _envBin

reload(_envBin)

import Binary._agent_MTB_binary as _agent_MTB_binary
import Binary._agent_GLB as _agent_GLB
import Binary._agent_TS_binary as _agent_TS_binary
import Binary._agent_meta_TS_binary as _agent_meta_TS_binary

reload(_agent_MTB_binary)
reload(_agent_GLB)
reload(_agent_TS_binary)
reload(_agent_meta_TS_binary)



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
class Experiment():
    """ 
    main module for running the experimnet
    """
    @autoargs()
    def __init__(self, N, T, K, p_raw, p, sigma, Sigma_delta
                 , u_theta, Sigma_theta, phi_beta = None, is_Binary = False
                 , order = "episodic"
                , seed = 42
                , with_intercept = True, X_mu = None, X_Sigma = None
                , title_settting = None, date_time = None
                 , misspecification = None
                ):
        self.setting = locals()
        self.setting['self'] = None
        if is_Binary:
            self.env = _envBin.Environment(N = N, T = T, K = K, p_raw = p_raw, p = p, phi_beta = phi_beta
                                              , u_theta = u_theta, Sigma_theta = Sigma_theta, seed = seed
                                          , with_intercept = with_intercept, X_mu = X_mu, X_Sigma = X_Sigma
                                           , misspecification = misspecification
                                          )
        else:
            self.env = _env.Environment(N = N, T = T, K = K, p_raw = p_raw, p = p, sigma = sigma, Sigma_delta = Sigma_delta
                                                  , u_theta = u_theta, Sigma_theta = Sigma_theta, seed = seed
                                       , with_intercept = with_intercept, X_mu = X_mu, X_Sigma = X_Sigma
                                        , misspecification = misspecification
                                       )
        self.theta = self.env.theta
        self.Xs = self.env.Phi # [N, K, p]
        self.r = self.env.r
        self.get_task_sequence()
   
    def _init_agents(self, agents = None):
        # sigma, Sigma_delta
        self.agents = agents
        self.agent_names = agent_names = list(agents.keys())
        self.record = {}
        self.record['R'] = {name : [] for name in agent_names}
        self.record['R']['oracle'] = []
        self.record['A'] = {name : [] for name in agent_names}
        self.record['regret'] = {name : [] for name in agent_names}
        self.record['meta_regret'] = {name : [] for name in agent_names}


    def get_task_sequence(self):
        self.task_sequence = []
        if self.order == "episodic":
            for i in range(self.N):
                for t in range(self.T):
                    self.task_sequence.append([i, t])
        if self.order == "concurrent":
            for t in range(self.T):
                for i in range(self.N):
                    self.task_sequence.append([i, t])
       
        
    def run(self):
        # sample one task, according to the order
        self.run_4_one_agent('oracle')
        for name in self.agent_names: 
            self.run_4_one_agent(name)
        self.post_process()

    def run_4_one_agent(self, name):
        self.j = -1
        if name in ["MTB", "GLB-TS", "MTB-approx"]: # 
            if self.seed % 16 == 0:
                for i, t in tqdm(self.task_sequence, desc = name
                                 , position=0
                                 , mininterval = 30
                                , miniters = 20):
                    self.j += 1
                    self.run_one_time_point(i, t, name)
                    if i % 10 == 0 and t % 10 == 0:
                        with open('log/{}.txt'.format(self.date_time), 'a') as f:
                            print("{} : seed {}, task {}, iter {}".format(name, self.seed, i, t), file=f)
            else:
                for i, t in self.task_sequence:
                    self.j += 1
                    self.run_one_time_point(i, t, name)
            with open('log/{}.txt'.format(self.date_time), 'a') as f:
                print("{} : seed {} DONE!".format(name, self.seed), file=f)

        else:
            for i, t in self.task_sequence:
                self.j += 1
                self.run_one_time_point(i, t, name)

    def run_one_time_point(self, i, t, name):
        if name == "oracle":
            R_opt = self.env.get_optimal_reward(i, t)
            self.record['R']["oracle"].append(R_opt)
        else:
            X = self.Xs[i]
            # provide the task id and its feature to the agent, and then get the action from the agent
            A = self.agents[name].take_action(i, t, X)
            # provide the action to the env and then get reward from the env
            R = self.env.get_reward(i, t, A)
            # collect the reward
            self.record['R'][name].append(R)
            self.record['A'][name].append(A)
            # provide the reward to the agent
            if self.is_Binary:
                if name in ["MTB", "meta-TS"] and self.order == "episodic":
                    self.agents[name].receive_reward(i, t, A, R, X, t == self.T - 1)
                else:
                    self.agents[name].receive_reward(i, t, A, R, X)
            else:
                if name == "meta-TS" and self.order == "episodic":
                    self.agents[name].receive_reward(i, t, A, R, X, t == self.T - 1)
                else:
                    self.agents[name].receive_reward(i, t, A, R, X)
  
    def post_process(self):
        for name in self.agent_names:
            self.record['regret'][name] = arr(self.record['R']["oracle"]) - arr(self.record['R'][name])

        self.record['cum_regret'] = {name : np.cumsum(self.record['regret'][name]) for name in self.agent_names}
        # x: time, y: cum_regret: group, name
        self.record['cum_regret_df'] = self.organize_Df(self.record['cum_regret'])
                
        if "oracle-TS" in self.agent_names:
            for name in self.agent_names:
                self.record['meta_regret'][name] = arr(self.record['R']['oracle-TS']) - arr(self.record['R'][name])
            self.record['cum_meta_regret'] = {name : np.cumsum(self.record['meta_regret'][name]) for name in self.agent_names}
            self.record['cum_meta_regret_df'] = self.organize_Df(self.record['cum_meta_regret'])


    def organize_Df(self, r_dict):
        T = len(r_dict[self.agent_names[0]])
        a = pd.DataFrame.from_dict(r_dict)
        # a.reset_index(inplace=True)
        a = pd.melt(a)
        a['time'] = np.tile(np.arange(T), len(self.agent_names))
        a = a.rename(columns = {'variable':'method'
                           , "value" : "regret"
                           , "time" : "time"})
        return a

            
    def plot_regret(self, skip_methods = ["TS"], plot_meta = True):
        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])
        if plot_meta:
            data_plot =  self.record['cum_meta_regret_df'] 
            data_plot = data_plot[data_plot.method != "oracle-TS"]
        else:
            data_plot =  self.record['cum_regret_df'] 
        if skip_methods is not None:
            for met in skip_methods:
                data_plot = data_plot[data_plot.method != met]
        ax = sns.lineplot(data = data_plot
                     , x="time", y="regret"
                          , n_boot = 100
                     , hue="method" # group variable
                    )

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

class run_experiment():
    @autoargs()
    def __init__(self, N, T, sigma1_square = None, sigma = None, K = None, p  = None
                 , phi_beta = None, is_Binary = False
                , order = "episodic"
                 , print_SNR = True
                 , with_intercept = True
                 , save_prefix = None
                 , debug_MTB = False
                 , Sigma_theta_factor = 1, Sigma_x_factor = 1
                 , misspecification = None
                 , only_ratio = False   
                 , MTS_freq = 2
                 , GLB_freq = "auto"
                ):
        if GLB_freq == "auto":
            if self.order == "episodic":
                self.GLB_freq = T
            else:
                self.GLB_freq = N
        self.setting = locals()
        self.setting['self'] = None
        self.title_settting = " ".join([str(key) + "=" + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float]])
        printR(self.title_settting)
        self.date_time = get_date() + get_time()
        ################
        if Sigma_theta_factor == "1":
            self.Sigma_theta = np.identity(p)
        else:
            self.Sigma_theta  = np.identity(p) / p #np.sqrt(p)  # such a normalization is not good. Should be ||theta|| = 1
            self.Sigma_theta *= Sigma_theta_factor
        self.u_theta = np.zeros(p)
        self.X_mu = np.zeros(p)
        self.X_Sigma = identity(p) * Sigma_x_factor
        
        if is_Binary:
            self.names = ["OSFA", "MTB", "individual-TS", "oracle-TS", "GLB-TS", "meta-TS"]
        else:
            self.names = ["OSFA"
                          , "MTB", "MTB-approx"
                          , "individual-TS", "oracle-TS", "linear-TS", "meta-TS"]
        if is_Binary:
            self.get_ri_prior_mean_prec(K, phi_beta)
        else:
            self.Sigma_delta = np.identity(K) * sigma1_square
            self.determine_some_priors_4_Gaussian(p, K, sigma1_square)
    
    #############################################################################################################################################
    #############################################################################################################################################
    def determine_some_priors_4_Gaussian(self, p, K, sigma1_square):
        
        
        MC_thetas = np.random.multivariate_normal(self.u_theta, self.Sigma_theta, 1000)[:, K:]
        MC_thetas = np.mean(np.sum(MC_thetas ** 2, 0))
        
        var_TS = MC_thetas + 1 / p + sigma1_square
        self.cov_TS = var_TS * identity(K)
    
        random_effect_std = np.sqrt(self.sigma1_square)
        fixed_effect_std = np.sqrt(MC_thetas)
        
        if self.print_SNR:
            print("std(r_ia) : fixed_effect = {:.2f}, random_effect = {:.2f}\
            , random_effect / fixed_effect = {:.2f} \n ".format(mean(fixed_effect_std), mean(random_effect_std), mean(random_effect_std) / mean(fixed_effect_std)))

    def get_ri_prior_mean_prec_fixed_theta(self, K, phi_beta, u_theta, Sigma_theta, sample_theta, seed
                                           , n_rep = 500):
        """ get the prior mean and precision phi of ri for meta_TS
        """
        
        if self.with_intercept:
            intercept_Phi = np.repeat(identity(K)[np.newaxis, :, :], n_rep, axis = 0)
            random_part_Phi = np.random.multivariate_normal(self.X_mu[K:], self.X_Sigma[K:][:, K:], (n_rep, K))
            sample_PHI = np.concatenate([intercept_Phi, random_part_Phi], axis = 2)
        else:
            sample_PHI=np.random.multivariate_normal(self.X_mu, self.X_Sigma, (n_rep,K))

        
        temp=[Phi_i.dot(sample_theta) for Phi_i in sample_PHI]

        temp1=np.exp(temp)/(np.exp(temp)+1)
        E_ri_thetai = np.mean(temp1, 0)

        temp2 = np.exp(temp) / (np.exp(temp)+1)  / (np.exp(temp)+1) # * (1-np.exp(temp))
        c_b_i1 = np.mean(temp2, 0) * phi_beta / (1+phi_beta) # conditional on x
        c_b_i2 = np.var(temp1, 0) # caused by x
        cov_ri_thetai = c_b_i1 + c_b_i2 
        
        signal_noise = {
            "fixed_effect" : c_b_i2
            , "random_effect" :  c_b_i1
        }
        if self.print_SNR and seed <= 1:
            print("std(r_ia) : fixed_effect = {:.2f}, random_effect = {:.2f}, random_effect / fixed_effect = {:.2f} \n ".format(mean(np.sqrt(c_b_i2)), mean(np.sqrt(c_b_i1))
                                                                , mean(np.sqrt(c_b_i1)) / mean(np.sqrt(c_b_i2))))
        
        ri_prior_phi_MC_fixed_theta = meanvar_meanprex(E_ri_thetai, cov_ri_thetai)

        return E_ri_thetai, ri_prior_phi_MC_fixed_theta
       
    
    def get_ri_prior_mean_prec(self, K, phi_beta
                              , n_rep = 500):
        """ get the prior mean and precision phi of ri for ind_TS, OSFA
        """
        sample_thetas = np.random.multivariate_normal(self.u_theta, self.Sigma_theta, n_rep)
        if self.with_intercept:
            intercept_Phi = np.repeat(identity(K)[np.newaxis, :, :], n_rep, axis = 0)
            random_part_Phi = np.random.multivariate_normal(self.X_mu[K:], self.X_Sigma[K:][:, K:], (n_rep, K))
            sample_PHI = np.concatenate([intercept_Phi, random_part_Phi], axis = 2)
        else:
            sample_PHI = np.random.multivariate_normal(self.X_mu, self.X_Sigma, (n_rep, K))
        
        Conditional_Cov_ri=[]
        Conditional_Mu_ri=[]
        for sample_theta in sample_thetas:
            temp=[Phi_i.dot(sample_theta) for Phi_i in sample_PHI]
            
            temp1=np.exp(temp)/(np.exp(temp)+1)
            E_ri_thetai = np.mean(temp1, 0)
            
            temp2 = np.exp(temp) / (np.exp(temp)+1)  / (np.exp(temp)+1) # * (1-np.exp(temp))
            c_b_i1 = np.mean(temp2, 0) * phi_beta / (1+phi_beta)
            c_b_i2= np.var(temp1, 0)
            cov_ri_thetai=np.diag(c_b_i1)+np.diag(c_b_i2)
            Conditional_Mu_ri.append(E_ri_thetai)
            Conditional_Cov_ri.append(cov_ri_thetai)
        
        E_cond_cov_ri=np.mean(Conditional_Cov_ri,axis=0)
        Cov_cond_mu_ri=np.cov(np.array(Conditional_Mu_ri).T,bias=False)
        Cov_ri=E_cond_cov_ri+Cov_cond_mu_ri
        
        ### OSFA, individual-TS
        self.ri_prior_mean_MC = np.array([1/2]*K)
        self.ri_prior_phi_MC = meanvar_meanprex(self.ri_prior_mean_MC,np.diag(Cov_ri))
        
    #############################################################################################################################################
    #############################################################################################################################################        
    def run_one_seed_Binary(self, seed):
        N, T, phi_beta, K, p = self.N, self.T, self.phi_beta, self.K, self.p
        self.exp = Experiment(N = N, T = T, K = K, p_raw = p, p = p, sigma = None, Sigma_delta = None
                     , u_theta = self.u_theta
                    , Sigma_theta  = self.Sigma_theta # for normalization
                    , phi_beta = phi_beta, is_Binary = self.is_Binary
                     , order = self.order
                    , seed = seed
                      , with_intercept = self.with_intercept, X_mu = self.X_mu, X_Sigma = self.X_Sigma
                        , title_settting = self.title_settting, date_time = self.date_time
                 , misspecification = self.misspecification
                             )
        self.theta = self.exp.env.theta
        self.r = self.exp.env.r
        ###################################### Priors ##############################################################
        E_ri_thetai, ri_prior_phi_MC_fixed_theta = self.get_ri_prior_mean_prec_fixed_theta(K, phi_beta, self.u_theta, self.Sigma_theta, sample_theta = self.theta, seed = seed)
        if self.only_ratio: 
            return [None, None] 
        # mean and phi
        TS = _agent_TS_binary.TS_agent(u_prior_mean=self.ri_prior_mean_MC, prior_phi_beta=self.ri_prior_phi_MC) 
        N_TS = _agent_TS_binary.N_TS_agent(u_prior_mean=self.ri_prior_mean_MC, prior_phi_beta=self.ri_prior_phi_MC, N = N)

        meta_TS_agent = _agent_meta_TS_binary.meta_TS_agent(T = self.T, order = self.order
                                                     , theta_prior_mean = self.u_theta, theta_prior_cov = self.Sigma_theta
                                                     , u_prior_mean = E_ri_thetai, ri_prior_phi_MC_fixed_theta= ri_prior_phi_MC_fixed_theta
                                                     , phi_beta = phi_beta, Xs = self.exp.Xs # [N, K, p]
                                                     , N = N, K = K
                                                    , update_freq = N
                                                    , true_priors_4_debug = [self.exp.env.alpha_Beta, self.exp.env.beta_Beta]
                                                    )
        meta_oracle = _agent_TS_binary.meta_oracle_agent(u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta)
        
        
        alpha_practical = 1 
        alpha = alpha_practical

        GLB_agent = _agent_GLB.GLB_agent(N = N, K = K, p = p
                                        , alpha = alpha, retrain_freq = self.GLB_freq)

        if self.debug_MTB:
            MTB_agent = _agent_MTB_binary.MTB_agent(start="oracle_4_debug", phi_beta = phi_beta
                         , theta_prior_mean = self.u_theta, theta_prior_cov = self.Sigma_theta
                        , T = self.T, order = self.order
                         , Xs = self.exp.Xs # # [N, K, p]
                        , update_freq = self.MTS_freq
                        , true_theta_4_debug = self.exp.env.theta
                        , exp_seed = seed
                        , u_prior_alpha = self.exp.env.alpha_Beta, u_prior_beta = self.exp.env.beta_Beta
                        )
        else:
            MTB_agent = _agent_MTB_binary.MTB_agent(start="cold", phi_beta = phi_beta
                     , theta_prior_mean = self.u_theta, theta_prior_cov = self.Sigma_theta
                    , T = self.T, order = self.order
                     , Xs = self.exp.Xs # # [N, K, p]
                    , update_freq = self.MTS_freq
                    , true_theta_4_debug = self.exp.env.theta, true_r_4_debug = np.concatenate([r for r in self.exp.env.r]) #self.exp.env.r
                    , exp_seed = seed
                    )
           

        ####################################################################################################
        agents = {
            "OSFA" : TS
            , "MTB" : MTB_agent
            , "individual-TS" : N_TS
            , "oracle-TS" : meta_oracle
            , "GLB-TS": GLB_agent
            , "meta-TS" :  meta_TS_agent
        }
        self.exp._init_agents(agents)

        self.exp.run()
        self.record = {0: self.exp.record}
        self.agents = self.exp.agents
        if "MTB" in agents:
            return [self.exp.record, self.exp.agents['MTB'].recorder]
        else:
            return [self.exp.record, None]

        
    def run_one_seed_Gaussian(self, seed):
        N, T, sigma1_square, sigma, K, p = self.N, self.T, self.sigma1_square, self.sigma,self.K, self.p
        
        
        self.exp = Experiment(N = N, T = T, K = K, p_raw = p, p = p, sigma = sigma, Sigma_delta = self.Sigma_delta
                     , u_theta = self.u_theta
                    , Sigma_theta  = self.Sigma_theta # for normalization
                    , phi_beta = self.phi_beta, is_Binary = self.is_Binary
                     , order = self.order
                    , seed = seed
                      , with_intercept = self.with_intercept, X_mu = self.X_mu, X_Sigma = self.X_Sigma
                        , title_settting = self.title_settting, date_time = self.date_time
                        , misspecification = self.misspecification
                    )
        self.theta = self.exp.env.theta
        
        if self.only_ratio: 
            return [None, None] 
        ###################################### Priors ##############################################################
        TS = _agent_TS.TS_agent(sigma = sigma
                                , u_prior_mean = np.zeros(K), u_prior_cov = self.cov_TS) #  / 10000000
        N_TS = _agent_TS.N_TS_agent(sigma = sigma
                                      , u_prior_mean = np.zeros(K), u_prior_cov = self.cov_TS
                                      , N = N)
        ###########
        LB_agent = _agent_LB.LB_agent(sigma = np.sqrt(sigma ** 2 + self.sigma1_square)
                                      , prior_theta_u = self.u_theta, prior_theta_cov = self.Sigma_theta
                                      , N = N, K = K, p = p)
        meta_oracle = _agent_TS.meta_oracle_agent(sigma = sigma
                                                    , u_prior_mean = self.exp.env.r_mean, u_prior_cov = [self.Sigma_delta for _ in range(N)])


        meta_TS_agent = _agent_meta_TS.meta_TS_agent(sigma = sigma, order = self.order
                 , sigma1_square = self.sigma1_square, theta = self.exp.theta
                 , N = N, K = K, p = p)
        
        MTB_agent = _agent_MTB.MTB_agent(sigma = sigma, order=self.order
                     , theta_prior_mean = self.u_theta, theta_prior_cov = self.Sigma_theta
                    , T = self.T
                     , delta_cov = self.Sigma_delta #np.identity(K)
                     , Xs = self.exp.Xs # # [N, K, p]
                    , update_freq = self.MTS_freq # performance will degenerate
                    )
        
        MTB_agent_approx = _agent_MTB.MTB_agent(sigma = sigma, order=self.order
                     , theta_prior_mean = self.u_theta, theta_prior_cov = self.Sigma_theta
                    , T = self.T
                     , delta_cov = self.Sigma_delta #np.identity(K)
                     , Xs = self.exp.Xs # # [N, K, p]
                    , update_freq = self.MTS_freq # performance will degenerate
                    , approximate_solution = True
                    )

        ####################################################################################################
        agents = {
            "OSFA" : TS
            , "MTB" : MTB_agent
            , "MTB-approx": MTB_agent_approx
            , "individual-TS" : N_TS
            , "oracle-TS" : meta_oracle
            , "linear-TS" : LB_agent
            , "meta-TS" :  meta_TS_agent
        }
        self.exp._init_agents(agents)
        
        self.exp.run()  
        self.record = {0: self.exp.record}  
        self.agents = self.exp.agents   
        if "MTB" in agents: 
            return [self.exp.record, self.exp.agents['MTB'].recorder]  
        else:   
            return [self.exp.record, None] 
    #############################################################################################################################################
    #############################################################################################################################################
    
    def run_multiple_parallel(self, reps, batch = 1):
        rep = reps // batch
        with open('log/{}.txt'.format(self.date_time), 'w') as f:
            print(self.title_settting, file=f)

        import ray
        if self.is_Binary:
            ray.shutdown()
            @ray.remote(num_cpus = 6) # num_cpus = 3
            def one_seed(seed):
                os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
                os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
                os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
                os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
                os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
                r = self.run_one_seed_Binary(seed)
                return r
            ray.init()
            ###########
            futures = [one_seed.remote(j) for j in range(reps)]
            record = ray.get(futures)
            ray.shutdown()
        else:
            record = []
            for b in range(batch):
                print("batch = {}".format(b))
                r = parmap(self.run_one_seed_Gaussian, range(rep * b, rep * b + rep))
                record += r
            self.record = record
        if not self.only_ratio: 
            self.record = [r[0] for r in record]
            self.record_MTB = [r[1] for r in record]
        
    #############################################################################################################################################
    #############################################################################################################################################
    def plot_regret(self, skip_methods = ["OSFA"]
                    , ci = None, freq = 20
                   , plot_mean = False, skip = 2
                   , y_min = -1, y_max = None):
        from matplotlib.transforms import BlendedGenericTransform

        # https://seaborn.pydata.org/generated/seaborn.lineplot.html
        #ax.legend(['label 1', 'label 2'])
        self.fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))

        COLORS = sns.color_palette("Set2")
        palette = {name : color for name, color in zip(self.names, COLORS)}

        n_methods = 6

        reps = len(self.record)
        # stacked over seeds. for each seed, according to the orders 
        # how to select should depends on self.setting['order'] * purpose
        # and how to calculate the average. 
        # if episodic: T first, then N  (trend with N) -> MR
        # if concurrent: N first, then T (trend with T) -> BR
        # BR: trend with T
        # MR: trend with N
        # data_plot_BR.time = j: according to (i, t), not (t)

        data = pd.concat([self.record[seed]['cum_regret_df'] for seed in range(reps)])
        data_meta = pd.concat([self.record[seed]['cum_meta_regret_df'] for seed in range(reps)])
        data_plot_meta = data_meta[data_meta.method != "oracle-TS"]


        if self.setting['order'] == "episodic":
            data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['T']) + self.setting['T'] - 1]   
            data_plot_BR.time = np.tile(np.arange(0, self.setting['N'])
                                        , len(data_plot_BR) // self.setting['N'])
            data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['T']) + self.setting['T'] - 1] 
            data_plot_meta.time = np.tile(np.arange(0, self.setting['N'])
                                        , len(data_plot_meta) // self.setting['N'])


        elif self.setting['order'] == "concurrent":
            data_plot_BR = data.iloc[np.arange(0, len(data), step = self.setting['N']) + self.setting['N'] - 1]   
            data_plot_BR.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_BR) // self.setting['T'])
            data_plot_meta = data_plot_meta.iloc[np.arange(0, len(data_plot_meta), step = self.setting['N']) + self.setting['N'] - 1] 
            data_plot_meta.time = np.tile(np.arange(0, self.setting['T'])
                                        , len(data_plot_meta) // self.setting['T'])

        self.data_plot_BR_original = data_plot_BR.copy()
        self.data_plot_meta_original = data_plot_meta.copy()

        if plot_mean:
            data_plot_BR.regret = data_plot_BR.regret / (data_plot_BR.time + 1)
            data_plot_meta.regret = data_plot_meta.regret / (data_plot_meta.time + 1)

        if skip_methods is not None:
            for met in skip_methods:
                data_plot_BR = data_plot_BR[data_plot_BR.method != met]
                data_plot_meta = data_plot_meta[data_plot_meta.method != met]

        data_plot_BR = data_plot_BR[data_plot_BR.time >= skip]
        data_plot_meta = data_plot_meta[data_plot_meta.time >= skip]


        ax1 = sns.lineplot(data=data_plot_BR
                     , x="time", y="regret"
                     , hue="method" # group variable
                    , ci = ci # 95
                    , ax = ax1
                           , n_boot = 20
                    , palette = palette
                    )
        ax1.set(ylim=(y_min, y_max))


        ax1.set_title('Bayes regret')
        ax1.legend().texts[0].set_text("Method")

        ax2 = sns.lineplot(data=data_plot_meta
                     , x="time", y="regret"
                     , hue="method" # group variable
                    , ci = ci # 95
                    , ax = ax2
                           , n_boot = 20
                    , palette = palette
                    )
        ax2.set(ylim=(y_min, None))


        ax2.set_title('Multi-task regret')
        ax2.legend().texts[0].set_text("Method")
        self.fig.suptitle(self.title_settting, fontsize=12, y = 1.1)

        handles, labels = ax1.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='lower center', ncol = len(labels))
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        plt.show()

    #############################################################################################################################################
        #############################################################################################################################################
    def save(self, main_path = "res/", fig_path = "fig/", sub_folder = [], no_care_keys = []
            , only_plot_matrix = 1):
        """
        Since all results together seems quite large
        a['record'][0].keys() = (['R', 'A', 'regret', 'meta_regret', 'cum_regret', 'cum_meta_regret'])

        regret / R can almost derive anything, except for A

        The only thing is that, we may need to re-read them. Probably we need a function, to convert a "skim recorder" to a "full recorder", when we do analysis.
        """
        ########################################################
        date = get_date()

        result_path = main_path + date
        fig_path = fig_path + date
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        if self.is_Binary:
            aa = "Binary" 
        else:
            aa = "Gaussian"
        fig_path += "/" + aa 
        result_path += "/" + aa 
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        ########################################################
        if len(sub_folder) > 0:
            fig_path += "/" 
            result_path += "/"
            for key in sub_folder:
                fig_path += ("_" + str(key) + str(self.setting[key]))
                result_path += ("_" + str(key) + str(self.setting[key]))
                no_care_keys.append(key)
        no_care_keys.append('save_prefix')
        if not self.is_Binary:
            no_care_keys.append('GLB_freq')

        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        ############################
        if only_plot_matrix:
            record_R_only = {"data_plot_BR_original" : self.data_plot_BR_original
                            , "data_plot_meta_original" : self.data_plot_meta_original}
        else:
            record_R_only = {seed : self.record[seed]['R'] for seed in range(len(self.record))}

        r = {"setting" : self.setting
             , "record" : record_R_only
            , "name" : self.names}

        ############################
        path_settting = "_".join([str(key) + str(self.setting[key]) for key in self.setting if type(key) in [str, int] and type(self.setting[key]) in [str, int, float] and key not in no_care_keys])
        print(path_settting)
        if self.save_prefix:
            path_settting = path_settting + "-" + self.save_prefix

        ############################
        r_path = result_path + "/"  + path_settting
        fig_path = fig_path + "/"  + path_settting + ".png"
        print("save to {}".format(r_path))
        self.fig.savefig(fig_path)
        dump(r,  r_path)
