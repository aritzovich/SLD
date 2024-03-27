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

    def fit(self, X, Y):
        self.m, self.my, self.sy, self.s2y= self.getStats(X,Y)
        self.statsToParams(self.m, self.my, self.sy, self.s2y)

    def exponent(self, X):
        '''
        Compute the exponents that determine the class probabilities
        h(y|x) propto np.exp(self.exponent(x)[y])

        :param X: the array of unlabeled instances
        :return:
        '''
        if self.canonical:
            return self.alpha_y + X.dot(self.beta_y.transpose()) - self.A()
        else:
            return self.alpha_y + X.dot(self.beta_y.transpose())

    def A(self):
        '''
        The log partition function for the logistic regression in the canonical form of the exponential
        family (self.canonical=True) under the equivalence with Gaussian naive Bayes with homocedasticity assumption

        Note that the terms independent from y are canceled in the softmax

        -See https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions normal distribution
        -The parameters self.kappa (one for ech predictor) correspond to -1/sigma^2.
        :return:
        '''
        # TODO: exponential_family#table_of_distributions has an error en the log partition function using natural params
        #A= -np.sum(self.beta_y**2/(4 * self.kappa), axis=1)# - np.sum(1/2 * np.log(-2 * self.kappa)) <- in wikipedia
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
    def getNbProbs(self,X):
        '''
        The class conditional distribution for the Gaussian naive Bayes under homocedasticity. The probabilities
        are obtained by evaluating the conditional Gaussian distribution p(x,y) = p(y) prod_i p(x_i|y) and then
        p(y|x) is obtained using Bayes rule

        It is used to check the equivalence with logistic regression clasification rule
        TODO: Remove it when everything works

        :param X: Unlabeled instances
        :return: p(Y|X)
        '''

        m,n= X.shape
        pY= np.zeros((m,self.cardY))
        for y in range(self.cardY):
            try:
                pY[:,y]= multivariate_normal.pdf(X, mean=self.mu_y[y], cov= self.v)* self.py[y]
            except:
                pY[:, y] = multivariate_normal.pdf(X, mean=self.mu_y[y], cov=self.v) * self.py[y]
        #Apply the Bayes rule to p(y,x) to obtain p(y|x) for each x in X (normalize)
        pY/= np.repeat(np.sum(pY,axis=1), self.cardY).reshape((m, self.cardY))

        return pY
    def predict(self,X):
        '''
        return the class labels with maximum probability for the given unlableled instances
        :param X: unlabeled instances
        :return: an array with the labels with maximum probability
        '''
        return np.argmax(self.getClassProbs(X), axis=1)

    def copy(self):
        '''
        Returns a copy of the current classifier

        TODO:To be implemented
        '''
        None

    def initialization(self):
        '''
        Standard initialization for the logistic regression: all the parameters are set to zero

        Warning: this is incompatible with the canonical form
        :return:
        '''
        self.beta_y= np.zeros((self.cardY,self.n))
        self.alpha_y= np.zeros(self.cardY)

    def minimumSquare(self,X,y, esz= 1.0):
        '''
        The parametric initialization for the logistic regression
        :param X:
        :param Y:
        :param esz:
        :return:
        '''
        # Add bias term to features
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        y_oh = np.eye(self.cardY)[y]


        # Solve least squares problem
        w = np.linalg.lstsq(X_bias, y_oh, rcond=None)[0]
        self.alpha_y= w[-1,:]
        self.beta_y= w[:-1,:].transpose()



    def statsToParams(self, m, my, sy, s2y):
        '''
        Compute the parameters of the classifier given the input statistics (CondMoments)
        '''

        # Parameters for Gaussian nb under homocedasticity
        self.py= my/m
        self.mu_y= sy/np.repeat(my, self.n).reshape(sy.shape)

        # varianza
        # self.v= np.sum(s2y,axis=0)/m - (np.sum(sy,axis=0)/m)**2

        # Promediado de varianzas
        # s2y= s2y/np.tile(my[:,np.newaxis],(1,self.n)) - (self.mu_y**2)* np.tile(self.py[:,np.newaxis],(1,self.n))
        # self.v= self.py.transpose().dot(s2y)

        # max likel estimate: (sum_i x^2 - sum_y m_y mu_y^2)/m = sum_i x^2/m - sum_y p(y) mu_y^2
        # self.v= (np.sum(s2y,axis= 0) - np.sum(sy**2/np.tile(my[:,np.newaxis],(1,self.n)), axis= 0))/m
        self.v=  np.sum(s2y,axis=0)/m - self.py.transpose().dot(self.mu_y**2)
        self.standardToNatural()

    def standardToNatural(self):
        # params without A
        # dependent term x_i, beta_y_i= mu_y_i/sigma_i^2
        # TODO duda: puede que sea mu_r-mu_Y/ sqrt(s2)
        self.beta_y= self.mu_y/self.v
        # alfa_y= ln p(y)/p(r) + sum_i (mu_r^2-mu_y^2)/2sigma_i^2 and alpha_r= 0
        if self.canonical:
            self.alpha_y = np.log(self.py)
            # Para poder calcular la log partition funcion A
            self.kappa= -1/self.v
        else:
            # When logistic regression is not in the canonical form the log partition function is included
            # in the parameters associated to the marginal of the class distribution, log(p(Y)). Note that the terms
            # independent from y in the log partition funcion are canceled in the softmax
            self.alpha_y = np.log(self.py) - np.sum(self.mu_y**2/(2 * self.v),axis=1)

    def getStats(self,X,Y,esz= 0):
        '''
        Return the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        esz: equivalent sample size
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
        Gradient descent for the LogosticReg classifiers
        :param X: unlabeled instances
        :param Y: labels
        :param pY: probabilistic labels
        :param h: logisticReg classifier
        :param lr: learning rate
        :param canonical: the form of the LogisticReg classifier
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


    def riskDesc(self, X, Y, lr= 0.1, loss= np.inf, stc= True):
        '''

        :param X: Instances
        :param Y: Labels
        :param lr: learning rate
        :param stc: stochastic/deterministic 0-1 loss
        '''

        m= X.shape[0]

        # one-hot encoding of Y
        #oh = np.zeros((m, self.cardY))
        #oh[np.arange(m), Y] = 1
        oh= np.eye(self.cardY)[Y]

        if stc:
            pY = self.getClassProbs(X)
        else:
            pY =  np.zeros((m, self.cardY))
            pY[np.arange(m), self.predict(X)] = 1

        dif = pY - oh

        # compute the change on the statistics
        d_m, d_my, d_sy, d_s2y = self.getStats(X,dif, esz= 0)
        self.m -= lr * d_m
        self.my -= lr * d_my
        self.sy -= lr * d_sy
        self.s2y -= lr * d_s2y

        aux= lr
        while np.any(self.m<= 0) or np.any(self.my <=0) or np.any(self.s2y <= 0.1):
            self.getStats(X, dif, esz=0)
            self.m += aux * d_m
            self.my += aux * d_my
            self.sy += aux * d_sy
            self.s2y += aux * d_s2y

            #aux*= 0.5
            #self.m -= aux * d_m
            #self.my -= aux * d_my
            #self.sy -= aux * d_sy
            #self.s2y -= aux * d_s2y
            self.statsToParams(self.m, self.my, self.sy, self.s2y)
            actloss= np.average(- np.log(self.getClassProbs(X)[np.arange(m), Y]))
            #print("       " + str(np.min(self.s2y))+": "+str(actloss))
            if actloss< loss:
                return

        if np.any(self.m<= 0): raise ValueError("invalid m: "+str(self.m))
        elif np.any(self.my <=0): raise ValueError("invalid my: "+str(self.my))
        elif np.any(self.s2y <= 0): raise ValueError("invalid s2y: "+str(self.s2y))


        # update the parameters
        self.statsToParams(self.m, self.my, self.sy, self.s2y)

        aux= lr
        while np.any(self.py <= 0) or np.any(self.v <= 0):
            self.m += aux * d_m
            self.my += aux * d_my
            self.sy += aux * d_sy
            self.s2y += aux * d_s2y
            aux*= 0.5
            self.m -= aux * d_m
            self.my -= aux * d_my
            self.sy -= aux * d_sy
            self.s2y -= aux * d_s2y
            self.statsToParams(self.m, self.my, self.sy, self.s2y)

        if np.any(self.py <= 0): raise ValueError("invalid py: "+str(self.py))
        elif np.any(self.v <= 0):

            raise ValueError("invalid v: "+str(self.v))