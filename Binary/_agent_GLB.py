from _util import *
from sklearn.linear_model import LogisticRegression
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.linear_model import SGDClassifier

class GLB_agent():
    """ Randomized exploration in generalized linear bandits
    """
    @autoargs()
    def __init__(self, N = 100, K = 2, p = 3
                , alpha = 1 # same with the paper
                 , true_theta_4_debug = None
                 , retrain_freq = 1
                ):
        
        self.K = K
        self.cnts = np.zeros((N, self.K))
        self.Xs = []
        self.XX = []
        self.Ys = []
        self.seed = 42
        self.clf = SGDClassifier(max_iter=1000, tol=1e-3, loss = "log", fit_intercept = False, random_state = 42, warm_start = True)

        self.theta_mean = ones(p)
        self.H = identity(p)
        self.theta_acc = []
        self.time_cost = {"inv" : 0, "sample" : 0, "other1" : 0, "other2" : 0}
        self.random_exploration = 0
        self.cnt = 0
        
    def derivative_logistic(self, x):
        num = np.exp(-x)
        return num / (1 + num) ** 2
        
    def take_action(self, i, t, X):
        """
        X = [K, p]
        """
        np.random.seed(self.seed)
        self.seed += 1
        try:
            self.inv_H = inv(self.H)
            self.sampled_theta = np.random.multivariate_normal(self.theta_mean, self.alpha ** 2 * self.inv_H)
            self.sampled_Rs = X.dot(self.sampled_theta) # monotone. logistic
            self.A = np.argmax(self.sampled_Rs)
        except:
            self.A = choice(self.K)
            self.random_exploration += 1
        
        return self.A
        
    def receive_reward(self, i, t, A, R, X):
        x = X[A]
        self.Xs.append(x)
        self.Ys.append(R)
        self.XX.append(np.outer(x, x))
        if len(set(self.Ys)) > 1 and self.cnt % self.retrain_freq == 0:
            self.clf.fit(self.Xs, self.Ys)
            self.theta_mean = self.clf.coef_[0]
            self.weights = self.derivative_logistic(arr(self.Xs).dot(self.theta_mean))
            self.H = arr(self.XX).T.dot(self.weights)

        if t % 100 == 0 and self.true_theta_4_debug is not None and i % 10 == 0:
            self.theta_acc.append(self.RMSE_theta(self.theta_mean))
            display(self.theta_acc)

        self.cnt += 1
    def RMSE_theta(self, v):
        return np.sqrt(np.mean((v - self.true_theta_4_debug) **2))
