import numpy as np
from scipy.stats import multivariate_normal



def project_onto_simplex(v):
    """
    Project a vector v onto the simplex with sum equal to s.

    Parameters:
    - v: Input vector to be projected

    Returns:
    - Projected vector onto the simplex
    """

    n = len(v)  # Number of elements in the input vector
    u = np.sort(v)[::-1]  # Sort the elements of v in descending order
    cssv = np.cumsum(u) - 1  # Compute the Cumulative Sum of Sorted Values (CSSV)
    ind = np.arange(1, n + 1)  # Indices from 1 to n
    cond = u - cssv / ind > 0  # Check the condition for each element
    rho = ind[cond][-1]  # Find the index where the condition is satisfied
    theta = cssv[cond][-1] / float(rho)  # Compute theta
    w = np.maximum(v - theta, 0)  # Project v onto the simplex
    return w


class LinearClassifier:
    def __init__(self, num_features, num_classes):
        self.num_classes = num_classes
        self.num_features = num_features
        self.weights = np.zeros((num_features, num_classes))

    def fit(self, X_train, y_train):
        '''
        Fitting by minimum squares
        :param X_train:
        :param y_train:
        :return:
        '''
        # Add bias term to features
        X_train_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        if y_train.ndim==1:
            # One-hot encoding of labels
            y_train_one_hot = np.eye(self.num_classes)[y_train]


        # Solve least squares problem
        self.weights = np.linalg.lstsq(X_train_bias, y_train_one_hot, rcond=None)[0]

    def riskDesc(self,X_train, y_train, lr= 0.1):

        diff= self.getClassProbs(X_train) - np.eye(self.num_classes)[y_train]
        X_train_bias = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        d_weights = np.linalg.lstsq(X_train_bias, diff, rcond=None)[0]

        self.weights-= lr * d_weights


    def getClassProbs(self, X):
        # Add bias term to features
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        # Predict probabilities for each class
        logits = np.dot(X_bias, self.weights)
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return probabilities

    def predict(self,X):
        # Predict class with highest probability
        return np.argmax(self.getClassProbs(X), axis=1)


class NaiveBayesDisc:
    def __init__(self, cardY, n, ess= 0):
        self.cardY = cardY
        self.num_features = n
        self.class_counts = np.zeros(cardY)
        self.feature_counts = np.zeros((n, 2, cardY))  # Stores counts for each feature
        self.class_probs = np.zeros(cardY)
        self.cond_probs = np.zeros((n, 2, cardY))
        self.ess= ess


    def initialize(self, ess=100):

        self.ess= ess
        self.feature_counts = np.ones((self.num_features, 2, self.cardY))* ess/(2*self.cardY)  # Stores counts for each feature
        self.class_counts= np.ones(self.cardY)* ess/self.cardY
        self.statsToParams()


    def getStats(self, X, Y):

        m,n= X.shape
        class_counts = self.ess * np.ones(self.cardY)/self.cardY
        feature_counts = self.ess * np.ones((n, 2, self.cardY))/(self.cardY*2)
        if Y.ndim == 1:
            # Compute class counts
            for c in range(self.cardY):
                class_counts[c] += np.sum(Y == c)

            # Compute feature counts
            for i in range(len(Y)):
                c = Y[i]
                for j in range(self.num_features):
                    try:
                        feature_counts[j, X[i, j], c] += 1
                    except:
                        feature_counts[j, X[i, j], c] += 1
        else:
            # Compute class counts
            class_counts += np.sum(Y,axis= 0)

            # Compute feature counts
            for i in range(m):
                for c in range(self.cardY):
                    for j in range(self.num_features):
                        feature_counts[j, X[i, j], c] += Y[i,c]

        return class_counts, feature_counts

    def statsToParams(self):
        # Compute class probabilities
        self.class_probs = self.class_counts / np.sum(self.class_counts)

        # Compute conditional probabilities
        for j in range(self.num_features):
            for f in range(2):
                for c in range(self.cardY):
                    self.cond_probs[j, f, c] = self.feature_counts[j, f, c] / self.class_counts[c]

    def fit(self, X_train, y_train):
        self.class_counts,self.feature_counts= self.getStats(X_train, y_train)
        self.statsToParams()

    def riskDesc(self, X, Y, lr= 0.1, correct_negative_stats= False):

        oh = np.eye(self.cardY)[Y]
        pY = self.getClassProbs(X)

        dif = pY - oh

        # compute the change on the statistics
        d_class_counts, d_feature_counts = self.getStats(X, dif)

        self.class_counts -= lr * d_class_counts
        self.feature_counts -= lr * d_feature_counts

        if correct_negative_stats:
            if np.any(self.class_counts < 0):
                print("correction in class counts")
                # Project to the positive part maintaining the sum
                sum = np.sum(self.class_counts)
                self.class_counts = project_onto_simplex(self.class_counts / sum) * sum

            if np.any(self.feature_counts < 0):
                print("correction in feature counts")
                for j in range(self.num_features):
                    for y in range(self.cardY):
                        if np.any(self.feature_counts[j,:,y]<0):
                            # Project to the positive part maintaining the sum
                            sum= np.sum(self.feature_counts[j,:,y])
                            self.feature_counts[j,:,y]= project_onto_simplex(self.feature_counts[j,:,y]/sum)*sum


        if np.any(self.class_counts < 0):
            raise ValueError("negative class_counts: " + str(np.sort(self.class_counts.flatten())[:10]))
        if np.any(self.feature_counts < 0):
            raise ValueError("negative feature_counts: " + str(np.sort(self.feature_counts.flatten())[:10]))

        # update the parameters
        self.statsToParams()


    def gradDesc(self, X, Y, lr= 0.1):
        '''
        Gradient descent for the LogosticReg classifiers
        :param X: unlabeled instances
        :param Y: labels
        :return:
        '''

        m= X.shape[0]

        # one-hot encoding of Y
        oh = np.zeros((m, self.cardY))
        oh[np.arange(m), Y] = 1

        pY= self.getClassProbs(X)
        dif= pY - oh

        d_alpha_y= np.sum(dif, axis= 0)/m
        alpha_y= np.log(self.class_probs)
        alpha_y -= lr * d_alpha_y
        self.class_probs= np.exp(alpha_y)
        # Normalize the probability table by projecting into the simplex
        self.class_probs= project_onto_simplex(self.class_probs)


        d_beta_y= np.zeros(2)
        for j in np.arange(self.num_features):
            for c in np.arange(self.cardY):
                # For each probability table: feature k and class c
                for k in np.arange(2):
                    d_beta_y[k]= np.sum(dif[X[:,j]== k,c])/m

                beta_y= np.log(self.cond_probs[j,:,c])
                beta_y -= lr * d_beta_y
                # Normalize each probability table by projecting into the simplex
                self.cond_probs[j, :, c]= project_onto_simplex(np.exp(beta_y))



    def getClassProbs(self, X):
        m,n= X.shape
        pY= np.zeros((m, self.cardY))
        for y in np.arange(self.cardY):
            pY[:, y]= self.class_probs[y]
            for j in np.arange(n):
                pY[:, y] *= self.cond_probs[j, X[:,j], y]

        row_sums = np.sum(pY, axis=1, keepdims=True)
        return pY/row_sums

    def predict(self, X):
        pY= self.getClassProbs(X)
        return np.argmax(pY, axis= 1)


class QDA:
    def __init__(self, cardY, n):
        self.cardY= cardY
        self.n= n

        self.m_y= None
        self.s1_y= None
        self.s2_y= None
        self.p_y = None
        self.mean_y = None
        self.cov_y = None

    def getStats(self, X, Y, ):

        m_y = np.zeros(self.cardY)
        s1_y = np.zeros((self.cardY, self.n))
        s2_y = np.zeros((self.cardY, self.n, self.n))

        if Y.ndim == 1:
            # Compute class counts
            for c in range(self.cardY):
                m_y[c] = np.sum(Y == c)

            # Compute con moments1
            for c in np.arange(self.cardY):
                X_c= X[Y==c,:]
                s1_y[c,:]=np.sum(X_c, axis= 0)
                s2_y[c, :]= X_c.transpose() @ X_c

        else:
            # Compute class counts
            m_y = np.sum(Y, axis=0)

            for c in np.arange(self.cardY):
                s1_y[c,:]= X.transpose() @ Y[:,c]
                s2_y[c, :]= (X.transpose() * Y[:,c]) @ X

        return m_y, s1_y, s2_y

    def statsToParams(self):

        self.p_y= self.m_y/np.sum(self.m_y)
        self.mean_y= np.zeros((self.cardY,self.n))
        self.cov_y= np.zeros((self.cardY,self.n,self.n))
        for c in np.arange(self.cardY):
            self.mean_y[c,:]= self.s1_y[c,:]/self.m_y[c]
            self.cov_y[c,:,:]= self.s2_y[c,:,:]/self.m_y[c] - np.outer(self.mean_y[c,:],self.mean_y[c,:])

    def fit(self,X,Y):
        self.m_y, self.s1_y, self.s2_y= self.getStats(X,Y)
        self.statsToParams()

    def getClassProbs(self, X):
        m= X.shape[0]
        pY= np.zeros((m,self.cardY))
        for c in np.arange(self.cardY):
            mvn = multivariate_normal(mean=self.mean_y[c,:], cov=self.cov_y[c,:,:], allow_singular= True)
            pY[:,c]= mvn.pdf(X)*self.p_y[c]

        pY/= np.sum(pY, axis= 1, keepdims= True)
        return pY

    def riskDesc(self, X, Y, lr= 0.1):

        oh = np.eye(self.cardY)[Y]
        pY = self.getClassProbs(X)

        dif = pY - oh

        # compute the change on the statistics
        d_m_y, d_s1_y, d_s2_y = self.getStats(X, dif)
        self.m_y -= lr * d_m_y
        self.s1_y -= lr * d_s1_y
        self.s2_y -= lr * d_s2_y

        # update the parameters
        self.statsToParams()

    def predict(self, X):
        pY= self.getClassProbs(X)
        return np.argmax(pY, axis= 1)
