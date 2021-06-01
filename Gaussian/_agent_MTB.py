from _util import *
import scipy.linalg
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

""" 
can still be largely save, by using those block structure.
"""

class MTB_agent():
    @autoargs()
    def __init__(self, sigma = 1, order="episodic",T = None
                 , theta_prior_mean = None, theta_prior_cov = None
                 , delta_cov = None
                 , Xs = None # [N, K, p]
                 , update_freq = 1
                 , approximate_solution = False
                ):
        self.K = K = len(delta_cov)
        self.N = N = len(Xs)
        self.p = p = len(theta_prior_mean)
        
        self.cnts = np.zeros((N, self.K))
        self.Phi = Xs # [N, K, p]
        self.Phi_i_Simga = [self.Phi[i].dot(theta_prior_cov) for i in range(N)]        

        self._init_data_storate(self.K, p, N)
        self._init_else(self.K, p, N)
        self._init_Sigma_12_part()
        self.seed = 42
        self.recorder = {}
        
        self.time_record = {name : 0 for name in ["compute_inverse", "update_posterior", "update_Sigma_12_part", "update_K_delta_Cov", "inv_inner", "inv_block2", "1", "2", "3", "update_Phi", "4", "collect_4_Sigma_12_part", "update_centered_R"] + list(np.arange(10,100))}
        self.theta_prior_cov_inv = inv(theta_prior_cov)
        self.tt = 0
        
        ## approx
        self.posterior_u_num, self.posterior_u_den, self.posterior_u, self.posterior_cov_diag = {}, {}, {}, {}

    def _init_else(self, K, p, N):
        # the two prior terms in the posterior
        if self.order == "concurrent" and self.approximate_solution:
            self.sampled_theta = np.random.multivariate_normal(self.theta_prior_mean, self.theta_prior_cov)
            self.ri_mean_prior = [self.Xs[i].dot(self.sampled_theta) for i in range(N)]
            self.ri_cov_prior = [self.delta_cov for i in range(N)]
    
            self.post_ri_num_wo_prior = [zeros(K) for i in range(N)]
            self.post_ri_den_wo_prior = [zeros(K) for i in range(N)]
            
            self.ri_num_post0 = [self.ri_mean_prior[i]/np.diag(self.delta_cov) for i in range(N)]
            self.ri_den_post0 = [1 / np.diag(self.delta_cov) for i in range(N)]
            
            self.ri_num_post = [self.ri_num_post0[i] + self.post_ri_num_wo_prior[i] for i in range(N)]
            self.ri_den_post = [self.ri_den_post0[i] + self.post_ri_den_wo_prior[i] for i in range(N)]
            
            self.ri_mean_post = [self.ri_num_post[i]/self.ri_den_post[i] for i in range(N)]
            self.ri_cov_post = [1/self.ri_den_post[i] for i in range(N)]
        else:
            self.ri_mean_prior = [self.Xs[i].dot(self.theta_prior_mean) for i in range(N)]
            self.ri_cov_prior = [self.Xs[i].dot(self.theta_prior_cov).dot(self.Xs[i].T) + self.delta_cov for i in range(N)]
    
            self.ri_mean_post = [self.Xs[i].dot(self.theta_prior_mean) for i in range(N)]
            self.ri_cov_post = [self.Xs[i].dot(self.theta_prior_cov).dot(self.Xs[i].T) + self.delta_cov for i in range(N)]

             
        self.K_first_term = None
        
        # to save storage so as to save time
        self.inv = {}
    
    def _init_empty(self, aa = None, p = None):
        K = self.K
        N = self.N
        if aa == "num":
            return [[0 for a in range(K)] for _ in range(N)]
        if aa == "col":
            return [np.zeros((K, 0)) for i in range(N)] 
        if aa == "row":
            return [[np.zeros((0, p)) for a in range(K)] for i in range(N)] 
        if aa == "mat_T":
            return [[np.zeros((self.T, self.T)) for a in range(K)] for i in range(N)]
        if aa == "null_scalar":
            return [[np.zeros(0) for a in range(K)] for i in range(N)]
        return [[np.zeros((0, 0)) for a in range(K)] for i in range(N)]
    
    def _init_data_storate(self, K, p, N):
        """ initialize data storage and components required for computing the posterior
        """
        if self.order == "concurrent" and self.approximate_solution:
            self.R_each_task = [[np.zeros(0) for a in range(self.K)] for _ in range(N)]
            self.origin_R = np.zeros(0)
        else:
            self.centered_R_each_task = [[np.zeros(0) for a in range(self.K)] for _ in range(N)]
            self.centered_R = np.zeros(0)
            
        self.next_pos_4_R_each_task = np.repeat(0, N * K)
        self.R_4_each_task = [[] for i in range(N)]
        
        self.observed_Xs = np.zeros((0, p))
        self.A_4_each_task = self._init_empty("num") #[zeros(K) for i in range(N)] #[[] for i in range(N)]
        
        self.Phi_obs = self._init_empty("row", p)
        self.Phi_obs_i = [np.zeros((0, p)) for i in range(N)]
        self.Phi_all = np.vstack(self.Phi_obs)
        
        ### Sigma_12_part
        # each i has a matrix
        self.Sigma12 = self._init_empty("col")
        # each i has a list, which stores the M_ij part for each j
        self.Sigma12_data = [[[np.zeros((K, self.T)) for a in range(K)] for j in range(N)] for i in range(N)]
        self.Sigma_idx = self._init_empty("num")
        ### inverse part
        self.J_inv_to_be_updated = set()
        self.solve2_part1_to_be_updated = [set() for _ in range(N)]
        self.solve2_part1_ij = [self._init_empty("row", K) for i in range(N)]
        self.K_delta_Cov_each_task = self._init_empty("mat_T") #[np.zeros((0, 0)) for i in range(N)]
        self.K_delta_Cov_each_task_sigmaI_inv = self._init_empty()

        self.J_sigmaI_inv_dot_Phi_each = self._init_empty("row", p)
        self.J_sigmaI_inv_dot_R_each = self._init_empty('null_scalar')
        self.J_sigmaI_inv_dot_Phi_R_each = self._init_empty()

       
    def _init_Sigma_12_part(self):
        self.Phi_i_Simga_dot_X = [[] for _ in range(self.N)]
        for ii in range(self.N):
            for i in range(self.N):
                X = self.Xs[i]
                a = self.Phi_i_Simga[ii].dot(X.T)
                self.Phi_i_Simga_dot_X[ii].append(a.T)
                if i == ii:
                    for A in range(self.K):
                        self.Phi_i_Simga_dot_X[ii][i][A] += self.delta_cov[A]

    ################################################################################################################################################
    ################################################################################################################################################
    def receive_reward(self, i, t, A, R, X):
        """update_data
        """
        if self.order == "concurrent" and self.approximate_solution:
            x = X[A]
            self.cnts[i, A] += 1
            a = now()
            self.update_Phi(i, A, x)
            self.time_record["update_Phi"] += (now() - a); a = now()
            
            self.update_K_delta_Cov(i, A)
            self.update_origin_R(i, R, A)
    
            self.A_4_each_task[i][A] += 1
            #self.A_4_each_task[i].append(A)
            self.R_4_each_task[i].append(R)
            self.Sigma_idx[i][A] += 1
            
            self.post_ri_num_wo_prior[i][A] += (R / self.sigma ** 2)
            self.post_ri_den_wo_prior[i][A] += (1 / self.sigma ** 2)
            
            self.ri_num_post[i][A] = self.ri_num_post0[i][A] + self.post_ri_num_wo_prior[i][A]
            self.ri_den_post[i][A] = self.ri_den_post0[i][A] + self.post_ri_den_wo_prior[i][A]
            
            self.ri_mean_post[i][A] = self.ri_num_post[i][A]/self.ri_den_post[i][A]
            self.ri_cov_post[i][A] = 1/self.ri_den_post[i][A]
        else:
            x = X[A]
            self.cnts[i, A] += 1
            a = now()
            self.update_Phi(i, A, x)
            self.time_record["update_Phi"] += (now() - a); a = now()
            
    
            self.collect_4_Sigma_12_part(i, R, A, x)
            self.time_record["collect_4_Sigma_12_part"] += (now() - a); a = now()
    
            self.update_K_delta_Cov(i, A)
            self.time_record["update_K_delta_Cov"] += (now() - a); a = now()
            self.update_centered_R(i, R, A)
            self.time_record["update_centered_R"] += (now() - a); a = now()
    
            self.A_4_each_task[i][A] += 1
            self.R_4_each_task[i].append(R)
            self.Sigma_idx[i][A] += 1
            
            if self.approximate_solution:
                if t >= 1:
                    self.posterior_u_num[i][A] += (R / self.sigma ** 2)
                    self.posterior_u_den[i][A] += (1 / self.sigma ** 2)
                    self.posterior_u[i][A] = self.posterior_u_num[i][A] / self.posterior_u_den[i][A]
                    self.posterior_cov_diag[i][A] = 1 / self.posterior_u_den[i][A]

    def take_action(self, i, t, X):
        np.random.seed(self.seed)
        self.seed += 1
        a = now()
        if self.approximate_solution:
            if self.order == "episodic":
                if t >= 1:
                    if t == 1:
                        self.update_Sigma_12_part(i)
                        self.compute_inverse()
                        self.update_posterior(i)
                        self.init_prior_approx(i)
                    self.sampled_Rs = np.random.multivariate_normal(self.posterior_u[i], np.diag(self.posterior_cov_diag[i]))
                    self.A = np.argmax(self.sampled_Rs)
                    self.tt += 1
                    return self.A
            else:
                if (i + 1) % self.N == 0:
                    self.compute_inverse()
                    self.sample_theta()
                    self.update_concurrent_post(self.N)
                self.sampled_Rs = np.random.multivariate_normal(self.ri_mean_post[i], np.diag(self.ri_cov_post[i]))
                self.A = np.argmax(self.sampled_Rs)
                self.tt += 1
                return self.A
        elif self.tt >= 1 and (t < 10 or t % self.update_freq == 0): 
            self.update_Sigma_12_part(i)
            self.time_record["update_Sigma_12_part"] += (now() - a); a = now()
            self.compute_inverse()
            self.time_record["compute_inverse"] += (now() - a); a = now()
            self.update_posterior(i)
            self.time_record["update_posterior"] += (now() - a); a = now()
            
        self.sampled_Rs = np.random.multivariate_normal(self.ri_mean_post[i], self.ri_cov_post[i])
        self.A = np.argmax(self.sampled_Rs)
        self.tt += 1
        return self.A

    ################################################################################################################################################
    ###################################################### receive_reward #################################################################
    ################################################################################################################################################
    def update_Phi(self, i, A, x):
        """
        """
        self.Phi_obs[i][A] = self.vstack([self.Phi_obs[i][A], x])
        if self.Phi_obs[i][A].ndim == 1:
            self.Phi_obs[i][A] = self.Phi_obs[i][A][np.newaxis, :]
        
        self.Phi_obs_i[i] = np.vstack(self.Phi_obs[i])
        self.Phi_all = np.vstack(self.Phi_obs_i)
        
    def collect_4_Sigma_12_part(self, i, R, A, x):
        idx = self.Sigma_idx[i][A]
        for ii in range(self.N):
            self.Sigma12_data[ii][i][A][:, idx] = self.Phi_i_Simga_dot_X[ii][i][A] #new_col
        

    def update_K_delta_Cov(self, i, A):
        """ the second term 
        """
        idx = self.Sigma_idx[i][A]
        num = self.delta_cov[A, A]
        to_add = np.repeat(num, self.A_4_each_task[i][A] + 1)

        self.K_delta_Cov_each_task[i][A][idx, :(idx + 1)] = to_add
        self.K_delta_Cov_each_task[i][A][:(idx + 1), idx] = to_add
        self.J_inv_to_be_updated.add((i, A))
        
        for j in range(self.N):
            self.solve2_part1_to_be_updated[j].add((i, A))

    def update_centered_R(self, i, R, A):
        this_R = R - self.Xs[i][A].dot(self.theta_prior_mean)
        self.centered_R_each_task[i][A] = np.append(self.centered_R_each_task[i][A], this_R)
        
        ik = i * self.K + A
        pos = self.next_pos_4_R_each_task[ik]
        self.centered_R = np.insert(self.centered_R, pos, this_R)
        self.next_pos_4_R_each_task[ik:] += 1
        

    ################################################################################################################################################
    ################################################################################################################################################
    def update_Sigma_12_part(self, i):        
        self.Sigma12[i] = []
        for j in range(self.N):
            for A in range(self.K):
                idx = self.Sigma_idx[j][A]
                a = self.Sigma12_data[i][j][A][:, :idx] 
                if a.shape[1] == 1:
                    a = np.squeeze(a)
                self.Sigma12[i].append(a)        
        
        self.Sigma12[i] = np.column_stack(self.Sigma12[i])
        if self.Sigma12[i].shape[1] == 1:
            self.Sigma12[i] = np.squeeze(self.Sigma12[i])
            
    def fast_inv(self, l):
        sigma, sigma_delta = self.sigma, np.sqrt(self.delta_cov[0, 0])
        return sigma ** -2 * identity(l) - sigma ** -4 * (sigma_delta ** -2 + sigma ** -2 * l) ** -1
    
    def vstack_list_of_list(self, list_of_list):
        return np.vstack([np.vstack([a for a in lis]) for lis in list_of_list])
    
    def conca_list_of_list(self, list_of_list):
        return np.concatenate([np.concatenate([a for a in lis ]) for lis in list_of_list])
    

    def compute_inverse(self):
        """ 
        (J_ia + sigma I)^{-1}
        = sigma ** -2 * identity(N_ia) - sigma ** -4 * (sigma1 ** -2 + sigma ** -2 * N_ia) ** -1 * 11'
        """
        a = now()
        if self.order == "concurrent" and self.approximate_solution:
            for i, A in self.J_inv_to_be_updated:
                N_i = j = self.Sigma_idx[i][A]
                aa = self.K_delta_Cov_each_task[i][A][:(j), :(j)]
                self.K_delta_Cov_each_task_sigmaI_inv[i][A] = self.fast_inv(j) 
                self.J_sigmaI_inv_dot_Phi_each[i][A] = self.K_delta_Cov_each_task_sigmaI_inv[i][A].dot(self.Phi_obs[i][A])
                self.J_sigmaI_inv_dot_R_each[i][A] = self.K_delta_Cov_each_task_sigmaI_inv[i][A].dot(self.R_each_task[i][A])
            self.inv['J_sigmaI_inv_dot_Phi_all'] = self.vstack_list_of_list(self.J_sigmaI_inv_dot_Phi_each)
            self.inv['J_sigmaI_inv_dot_R'] = self.conca_list_of_list(self.J_sigmaI_inv_dot_R_each)

        else:
            for i, A in self.J_inv_to_be_updated:
                N_i = j = self.Sigma_idx[i][A]
                aa = self.K_delta_Cov_each_task[i][A][:(j), :(j)]
                self.K_delta_Cov_each_task_sigmaI_inv[i][A] = self.fast_inv(j) #inv(aa + self.sigma ** 2 * np.identity(N_i))
                self.J_sigmaI_inv_dot_Phi_each[i][A] = self.K_delta_Cov_each_task_sigmaI_inv[i][A].dot(self.Phi_obs[i][A])
                self.J_sigmaI_inv_dot_R_each[i][A] = self.K_delta_Cov_each_task_sigmaI_inv[i][A].dot(self.centered_R_each_task[i][A])
                
            self.J_inv_to_be_updated = set()
            self.time_record["inv_inner"] += (now() - a); a = now()
    
            self.inv['J_sigmaI_inv_dot_Phi_all'] = self.vstack_list_of_list(self.J_sigmaI_inv_dot_Phi_each)

            ### Few time cost for steps below
            self.inv['inner_inv'] = inv(self.theta_prior_cov_inv + self.Phi_all.T.dot(self.inv['J_sigmaI_inv_dot_Phi_all']))        
            self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'] = self.inv['J_sigmaI_inv_dot_Phi_all'].dot(self.inv['inner_inv'])
            self.inv['J_sigmaI_inv_dot_R'] = self.conca_list_of_list(self.J_sigmaI_inv_dot_R_each)
            self.aa = self.inv['J_sigmaI_inv_dot_Phi_all'].T.dot(self.centered_R)
            self.inv_dot_R = self.inv['J_sigmaI_inv_dot_R'] - self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'].dot(self.aa)    
                


    def update_posterior(self, i):   
        sigma12 = self.Sigma12[i] # = self.Phi_i_Simga_theta_Phi_T[i] + self.Ms[i]
        try:
            a = now()
            # some time
            solve2_part2 = self.inv['J_sigmaI_inv_dot_Phi_all_dot_inner_inv'].dot(self.inv['J_sigmaI_inv_dot_Phi_all'].T.dot(sigma12.T))
            self.time_record["3"] += (now() - a); a = now()
            ## [1] only dominating for "concurrent"
            for j, A in self.solve2_part1_to_be_updated[i]:
                idx = self.Sigma_idx[j][A]
                self.solve2_part1_ij[i][j][A] = self.K_delta_Cov_each_task_sigmaI_inv[j][A].dot(arr(self.Sigma12_data[i][j][A][:, :(idx)].T))
            self.solve2_part1_to_be_updated[i] = set()
            self.time_record["inv_block2"] += (now() - a); a = now()
            # [2] some time
            solve2_part1 = self.vstack_list_of_list(self.solve2_part1_ij[i])
            solve2 = solve2_part1 - solve2_part2
            self.time_record["1"] += (now() - a); a = now()

            self.ri_mean_post[i] = self.ri_mean_prior[i] + sigma12.dot(self.inv_dot_R)
            self.ri_cov_post[i] = self.ri_cov_prior[i] - sigma12.dot(solve2)
            self.time_record["2"] += (now() - a); a = now()
    
        except:
            self.ri_mean_post[i] = self.ri_mean_prior[i] + sigma12 * self.inv_dot_R
            self.ri_cov_post[i] = self.ri_cov_prior[i] - np.outer(sigma12, sigma12) / 1 #self.Kernel_sigmaI

    def init_prior_approx(self, i):
        self.posterior_u_num[i] = self.ri_mean_post[i] / np.diag(self.delta_cov)
        self.posterior_u_den[i] = 1 / np.diag(self.delta_cov)
        self.posterior_u[i] = self.posterior_u_num[i] / self.posterior_u_den[i]
        self.posterior_cov_diag[i] = 1 / self.posterior_u_den[i]

            
    ################################################################################################################################################
    ################################################################################################################################################
    def hstack(self, C):
        A, B = C
        if A.ndim == 1:
            return np.hstack([A, B])
        elif A.shape[1] == 0:
            return B
        elif B.ndim == 1:
            return np.hstack([A, np.expand_dims(B, 1)])
        else:
            return np.hstack([A, B])
    
    def vstack(self, C):
        A, B = C
        if A.shape[0] == 0:
            return B
        else:
            return np.vstack([A, B])




    ################################################################################################################################################
    ####Concurrent case: sample theta first, then sample r; above are for the episodic setting
    #############################################################################################################################################
    def sample_theta(self):
        V_Phi = self.inv['J_sigmaI_inv_dot_Phi_all']
        V_R = self.inv['J_sigmaI_inv_dot_R']
        sigma_tilde= inv(self.Phi_all.T.dot(V_Phi)+inv(self.theta_prior_cov))#from update Phi
        mean = sigma_tilde.dot(self.Phi_all.T.dot(V_R)+self.theta_prior_cov_inv.dot(self.theta_prior_mean))
        self.sampled_theta = np.random.multivariate_normal(mean, sigma_tilde)
        

    def update_origin_R(self, i, R, A):
        this_R = R
        self.R_each_task[i][A] = np.append(self.R_each_task[i][A], this_R)
        
        ik = i * self.K + A
        pos = self.next_pos_4_R_each_task[ik]
        self.origin_R = np.insert(self.origin_R, pos, this_R)
        self.next_pos_4_R_each_task[ik:] += 1
    
    def update_concurrent_post(self,N):
        self.ri_mean_prior = [self.Xs[i].dot(self.sampled_theta) for i in range(N)]
        self.ri_cov_prior = [self.delta_cov for i in range(N)]
        
        self.ri_num_post0 = [self.ri_mean_prior[i]/np.diag(self.delta_cov) for i in range(N)]
        self.ri_den_post0 = [1 / np.diag(self.delta_cov) for i in range(N)]
            
        self.ri_num_post = [self.ri_num_post0[i] + self.post_ri_num_wo_prior[i] for i in range(N)]
        self.ri_den_post = [self.ri_den_post0[i] + self.post_ri_den_wo_prior[i] for i in range(N)]
        
        self.ri_mean_post = [self.ri_num_post[i]/self.ri_den_post[i] for i in range(N)]
        self.ri_cov_post = [1/self.ri_den_post[i] for i in range(N)]



    def print_time(self):
        total = sum(list(self.time_record.values()))
        a = {key : value for key, value in self.time_record.items() if value > total / 20}
        print(a)

