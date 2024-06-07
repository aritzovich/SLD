import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax

class LogReg:
    #TODO: lo de self.canonical deberia controlarse de forma interna. Si no hay estadisticos, no hace falta que sea en forma canonica
    def __init__(self,n,cardY,canonical= False):
        '''
        Constructor for the multiclass logistic regression classifier
        -All the variables are considered continuous and the feature vector corresponds to the identity, i.e.,phi(x)=(1,x)

        :param n: the number of predictor variables
        :param cardY: the number of class labels
        :param canonical: the model is given in the exponential canonical form. It includes the log partition function
        implemented in the procedure A. When canonical= False
        '''
        self.n= n
        self.cardY= cardY
        self.canonical= canonical

    def fit(self, X, Y, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0):
        '''
        Closed-form learning algorithm for LR

        Args:
            X: training unlabeled instances
            Y: training class labels
            ess:
            mean0:
            w_mean0:
            cov0:
            w_cov0:

        Returns:

        '''
        self.m, self.my, self.sy, self.s2y= self.getStats(X,Y)
        self.statsToParams(self.m, self.my, self.sy, self.s2y, ess, mean0, w_mean0, cov0, w_cov0)

    def exponent(self, X):
        '''
        Compute the exponents that determine the class probabilities
        h(y|x) propto np.exp(self.exponent(x)[y])

        :param X: unlabeled instances
        :return: the logarithm of the class probabilities for each unlabeled instance, log h(y|x)
        '''
        if self.canonical:
            return self.alpha_y + X.dot(self.beta_y.transpose()) - self.A()
        else:
            return self.alpha_y + X.dot(self.beta_y.transpose())

    def A(self):
        '''
        The log partition function for the logistic regression in the canonical form of the exponential
        family (self.canonical=True) under the equivalence with Gaussian naive Bayes with homocedasticity assumption

        Note that the terms independent from the class label are canceled in the softmax

        -See https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions normal distribution
        -The parameters self.kappa (one for ech predictor) correspond to -1/sigma^2.
        :return:
        '''
        A = -np.sum(self.beta_y ** 2 / (2 * self.kappa), axis=1)  # - np.sum(1/2 * np.log(-2 * self.kappa))
        return A

    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances
        '''

        if X.ndim>1:
            return softmax(self.exponent(X), axis=1)
        else:
            return softmax(self.exponent(X))

    def predict(self,X):
        '''
        return the class labels with maximum probability for the given unlabeled instances

        :param X: unlabeled instances
        :return: an array with the labels with maximum probability
        '''
        return np.argmax(self.getClassProbs(X), axis=1)

    def initialization(self):
        '''
        Standard initialization for the logistic regression: all the parameters are set to zero

        Warning: this is incompatible with the canonical form
        :return:
        '''
        self.beta_y= np.zeros((self.cardY,self.n))
        self.alpha_y= np.zeros(self.cardY)

    def statsToParams(self, m, my, sy, s2y, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0):
        '''
        Parameter mapping for the LR given the input statistics stored at attributes. Depending on the values of
        ess, w_mean0, w_cov0 maximum likelihood (values = 0) or maximum a posteriori (values > 0) parameters are
        computed

        Args:
            m: number of instances
            my: number of instances of class y
            sy: sum of x for the instances of class y
            s2y: sum of x^2 for the instances of class y
            ess: equivalent sample size of the Dirichlet uniform prior for the class marginal distribucion
            mean0: the prior mean vector for the normal-Whisart prior
            w_mean0: equivalent sample size for the Normal-Whisart prior of the mean vector conditioned to ech class
            label
            cov0: the variance prior for the normal-Whisart prior
            w_cov0: equivalent sample size for the Normal-Whisart prior of the variance

        Returns:

        '''

        # Parameters for Gaussian naive Bayes under homocedasticity
        if ess== 0:
            self.py = my / m
        else:
            m_y_prior = np.ones(self.cardY) * ess / self.cardY
            self.py= (my+ m_y_prior)/(m + ess)

        if w_mean0==0:
            self.mu_y= sy/np.repeat(my, self.n).reshape(sy.shape)
        elif w_mean0>0:
            prior_mean= np.ones((self.cardY, self.n))*mean0
            self.mu_y = (sy +prior_mean* w_mean0) /(np.repeat(my, self.n).reshape(sy.shape)+ w_mean0)

        if w_cov0==0:
            # s2/m - sum_y m_y/m * mu_y^2
            self.v=  np.sum(s2y,axis=0)/m - self.py.transpose().dot(self.mu_y**2)
        elif w_cov0>0:
            var= (np.sum(s2y,axis= 0) - np.sum(sy**2/np.tile(my[:,np.newaxis],(1,self.n)), axis= 0))/m
            prior_cov= np.ones(self.n)* cov0
            self.v= (var * m + prior_cov* w_cov0)/(m + w_cov0)

        self.standardToNatural()

    def standardToNatural(self):
        '''
        Transform the parameters to the exponential form of the model

        Depending on the form of the logistic regression (self.canonical) the log partition function (false) is included
        into the parameters associated to the class marginal distribution.
        '''
        self.beta_y= self.mu_y/self.v
        if self.canonical:
            self.alpha_y = np.log(self.py)
            # kappa is used to compute the log partition funcion, A
            self.kappa= -1/self.v
        else:
            # When logistic regression is not in the canonical form the log partition function is included
            # in the parameters associated to the marginal of the class distribution, log(p(Y)). The terms
            # independent from y in the log partition function are canceled
            self.alpha_y = np.log(self.py) - np.sum(self.mu_y**2/(2 * self.v),axis=1)

    def getStats(self,X,Y):
        '''
        Statistics mapping of the training data for LR

        Args:
            X: training unlabeled instances
            Y: training class labels

        return: number of instances, number of instances for each class, sum of x for each input feature conditioned to
        each class label, sum of x^2 for each input feature conditioned to each class label class
        '''

        if Y.ndim== 1:
            # Maximum likelihood statistics
            m= float(Y.shape[0])
            my= np.unique(Y, return_counts=True)[1].astype(float)
            sy= np.row_stack([np.sum(X[Y==y,:],axis= 0) for y in np.arange(self.cardY)])
            s2y= np.row_stack([np.sum(X[Y==y,:]**2,axis= 0) for y in np.arange(self.cardY)])
        elif Y.ndim== 2:
            # Weighted maximum likelihood statistics
            pY= Y
            m= np.sum(pY)
            my= np.sum(pY, axis= 0)
            sy= X.transpose().dot(pY).transpose()
            s2y= (X**2).transpose().dot(pY).transpose()

        return (m,my,sy,s2y)

    def gradDesc(self, X, Y, lr= 0.1):
        '''
        Gradient descent of the average negative log loss in training data for LR

        Args:
            X: training unlabeled instances
            Y: training class labels
            lr: learning rate (default 0.1)
        :return:
        '''

        m= X.shape[0]

        # one-hot encoding of Y
        oh = np.zeros((m, self.cardY))
        oh[np.arange(m), Y] = 1

        pY= self.getClassProbs(X)
        dif= pY - oh
        if self.canonical:
            d_alpha = np.average(dif, axis=0)
            d_beta = dif.transpose().dot(X)/m + np.tile(np.average(dif,axis= 0)[:,np.newaxis],(1,self.n)) *self.beta_y/(2*self.kappa)
            d_kappa = np.sum(np.tile(np.average(dif,axis= 0)[:,np.newaxis],(1,self.n)) *(-self.beta_y**2),axis= 0)/(4*self.kappa**2)
        else:
            d_alpha = np.average(dif, axis=0)
            d_beta = dif.transpose().dot(X) / m

        self.beta_y -= lr * d_beta
        self.alpha_y -= lr * d_alpha
        if self.canonical:
            self.kappa -= lr * d_kappa


    def riskDesc(self, X, Y, lr= 0.1, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0, correct_forbidden_stats= True):
        '''
        Risk-based calibration for LR. The closed from algorithm learn maximum likelihood (ess=0, w_mean0= 0,
        w_cov0= 0) or the maximum a posteriori parameters (values grater than 0)

        args:
            X: training unlabeled instances
            Y: training class labels
            lr: learning rate
            ess: equivalent sample size of the Dirichlet uniform prior for the class marginal distribucion
            mean0: the prior mean vector for the normal-Whisart prior
            w_mean0: equivalent sample size for the Normal-Whisart prior of the mean vector conditioned to ech class
            label
            cov0: the variance prior for the normal-Whisart prior
            w_cov0: equivalent sample size for the Normal-Whisart prior of the variance
            correct_forbidden_stats: when true (default) it does not update the statistics with negative statistics
        '''

        m= X.shape[0]

        # one-hot encoding of Y
        oh= np.eye(self.cardY)[Y]
        pY = self.getClassProbs(X)
        dif = pY - oh

        # compute the change on the statistics
        d_m, d_my, d_sy, d_s2y = self.getStats(X,dif)
        self.m -= lr * d_m
        self.my -= lr * d_my
        self.sy -= lr * d_sy
        self.s2y -= lr * d_s2y

        if correct_forbidden_stats:
            if np.any(self.my <= 0):
                self.my += lr * d_my

        self.statsToParams(self.m, self.my, self.sy, self.s2y, ess, mean0, w_mean0, cov0, w_cov0)

        if np.any(self.py <= 0): raise ValueError("invalid py: "+str(self.py))
        elif np.any(self.v <= 0): raise ValueError("invalid v: "+str(self.v))
