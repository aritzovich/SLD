import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax


def project_onto_simplex(v):
    """
    Project a vector v onto the simplex with sum equal to s.

    Parameters:
    - v: Input vector to be projected

    Returns:
    - Projected vector onto the simplex
    """

    try:
        n = len(v)  # Number of elements in the input vector
        u = np.sort(v)[::-1]  # Sort the elements of v in descending order
        cssv = np.cumsum(u) - 1  # Compute the Cumulative Sum of Sorted Values (CSSV)
        ind = np.arange(1, n + 1)  # Indices from 1 to n
        cond = u - cssv / ind > 0  # Check the condition for each element
        rho = ind[cond][-1]  # Find the index where the condition is satisfied
        theta = cssv[cond][-1] / float(rho)  # Compute theta
        w = np.maximum(v - theta, 0)  # Project v onto the simplex
        return w
    except Exception as e:
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
    def __init__(self, cardY, card, n):
        self.cardY = cardY
        self.num_features = n
        self.card= card
        self.class_counts = np.zeros(cardY)
        #self.feature_counts = np.zeros((n, 2, cardY))
        self.feature_counts = np.zeros((n, card, cardY))
        self.class_probs = np.zeros(cardY)
        #self.cond_probs = np.zeros((n, 2, cardY))
        self.cond_probs = np.zeros((n, card, cardY))

    def getStats(self, X, Y):
        m,n= X.shape
        class_counts = np.zeros(self.cardY)
        feature_counts = np.zeros((n, self.card, self.cardY))

        # Convert X to a boolean mask
        # X_mask = X.astype(bool)

        if Y.ndim == 1:
            # Compute class counts
            class_counts += np.bincount(Y, minlength=self.cardY)

            # Compute feature counts
            for j in range(self.num_features):
                feature_counts[j, :, :]= np.bincount(X[:,j]*self.cardY+ Y, minlength= self.card* self.cardY ).reshape((self.card,self.cardY))

                #feature_counts[j, :, :]= np.bincount(X[:,j]*self.cardY+ Y, minlength= 2* self.cardY ).reshape((2,self.cardY))
                #feature_counts[j, 0, :] += np.bincount(Y[~X_mask[:,j]], minlength=self.cardY)
                #feature_counts[j, 1, :] += np.bincount(Y[X_mask[:,j]], minlength=self.cardY)

        else:
            # Compute class counts
            class_counts += np.sum(Y,axis= 0)

            # Use broadcasting to update feature_counts efficiently
            for j in range(self.num_features):
                for x in range(self.card):
                    feature_counts[j, x, :] = np.sum(Y[X[:, j]==x,:], axis= 0)

                #feature_counts[j, 1, :]= X[:,j].dot(Y)
                #feature_counts[j, 0, :]= (1-X[:,j]).dot(Y)
                #feature_counts[j, 0, :] += np.sum(Y[~X_mask[:,j], :], axis=0)
                #feature_counts[j, 1, :] += np.sum(Y[X_mask[:,j], :], axis=0)

        return class_counts, feature_counts

    def statsToParams(self, ess= 0):
        '''
        Parameter mapping: from statistics to parameters.

        Args:
            ess: when zero corresponds to maximum likelihood parameters. When positive corresponds to maximum a
            posteriori parameters with a uniform dirichlet prior with ess equivalent sample size
        Returns:

        '''
        if ess>0:
            #feature_counts_prior = np.ones((self.num_features, 2, self.cardY)) * ess / (2 * self.cardY)
            feature_counts_prior = np.ones((self.num_features, self.card, self.cardY)) * ess / (self.card * self.cardY)
            class_counts_prior = np.ones(self.cardY) * ess / self.cardY

        # Compute class probabilities
        if ess==0:
            self.class_probs = self.class_counts / np.sum(self.class_counts)
        elif ess>0:
            self.class_probs = (self.class_counts + class_counts_prior) / np.sum(self.class_counts + class_counts_prior)

        # Compute conditional probabilities
        for j in range(self.num_features):
            #for f in range(2):
            for f in range(self.card):
                for c in range(self.cardY):
                    if ess== 0:
                        self.cond_probs[j, f, c] = self.feature_counts[j, f, c] / self.class_counts[c]
                    elif ess>0:
                        self.cond_probs[j, f, c] = (self.feature_counts[j, f, c] + feature_counts_prior[j, f, c]) / (
                                    self.class_counts[c] + class_counts_prior[c])

    def fit(self, X_train, y_train, ess= 0):
        self.class_counts,self.feature_counts= self.getStats(X_train, y_train)
        self.statsToParams(ess)

    def riskDesc(self, X, Y, lr= 0.1, ess= 0, correct_forbidden_stats= True):
        '''
        Risk descent iterative learning algorithm.

        Args:
            X: Training unlabeled instances
            Y: Training class labels
            lr: learning rate
            correct_negative_stats:
                1) project negative statistics by using the simplex projection of the associated probabilities
                2) do not update the table with negative statistics

        Returns:

        '''

        oh = np.eye(self.cardY)[Y]
        pY = self.getClassProbs(X)

        dif = pY - oh

        # compute the change on the statistics
        d_class_counts, d_feature_counts = self.getStats(X, dif)

        self.class_counts -= lr * d_class_counts
        self.feature_counts -= lr * d_feature_counts

        if correct_forbidden_stats:
            if np.any(self.class_counts < 0):
                self.class_counts -= lr * d_class_counts

            first, second= np.where((self.feature_counts < 0).any(axis=(1)))
            indices= list(zip(first, second))
            for j,y in indices:
                self.feature_counts[j, :, y] += lr * d_feature_counts[j, :, y]



        # update the parameters from the parameters
        self.statsToParams(ess)


    def gradDesc(self, X, Y, lr= 0.1):
        '''
        Gradient descent for the discrete naive Bayes classifiers. The gradient is computed for the parameters
        (probabilities) in their logarithm form to avoid negative parameters. Once the gradient descent is computed
        the parameters are projected into the simplex to guarantee that are probabilities

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


        d_beta_y= np.zeros(self.card)
        for j in np.arange(self.num_features):
            for c in np.arange(self.cardY):
                # For each probability table: feature k and class c
                for x in np.arange(self.card):
                    d_beta_y[x]= np.sum(dif[X[:,j]== x,c])/m

                beta_y = np.log(self.cond_probs[j,:,c])
                beta_y -= lr * d_beta_y
                # Normalize each probability table by projecting into the simplex
                self.cond_probs[j, :, c]= project_onto_simplex(np.exp(beta_y))



    def getClassProbs_(self, X):
        m,n= X.shape
        pY= np.zeros((m, self.cardY))
        for y in np.arange(self.cardY):
            pY[:, y]= self.class_probs[y]
            for j in np.arange(n):
                pY[:, y] *= self.cond_probs[j, X[:,j], y]

        row_sums = np.sum(pY, axis=1, keepdims=True)
        return pY/row_sums

    def getClassProbs(self, X):
        m,n= X.shape
        log_pY= np.zeros((m, self.cardY))
        for y in np.arange(self.cardY):
            log_pY[:, y]= np.log(self.class_probs[y])
            for j in np.arange(n):
                log_pY[:, y] += np.log(self.cond_probs[j, X[:,j], y])
        return softmax(log_pY, axis= 1)



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

    def getStats(self, X, Y):

        m_y = np.zeros(self.cardY)
        s1_y = np.zeros((self.cardY, self.n))
        s2_y = np.zeros((self.cardY, self.n, self.n))

        if Y.ndim == 1:
            # Compute class counts
            for c in range(self.cardY):
                m_y[c] = np.sum(Y == c)

            # Compute cond moments1
            for c in np.arange(self.cardY):
                X_c= X[Y==c, :]
                s1_y[c,:]=np.sum(X_c, axis= 0)
                s2_y[c, :]= X_c.transpose() @ X_c

        else:
            # Compute class counts
            m_y = np.sum(Y, axis=0)

            for c in np.arange(self.cardY):
                s1_y[c, :]= X.transpose() @ Y[:,c]
                s2_y[c, :]= (X.transpose() * Y[:,c]) @ X

        return m_y, s1_y, s2_y

    def statsToParams(self, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0):
        '''

        Args:
            ess: equivalent sample size for the Dirichlet prior of the class distribution
            mean0: Prior over the mean.
            w_mean0: Weight for the prior over the mean
            cov0: Prior for the variance.
            nu: Degrees of freedom for Wishart distribution
            scale: Scale matrix for Wishart distribution

        Returns:

        '''

        #Marginal probabilitiy class distribution
        m_y_prior= np.ones(self.cardY)*ess/self.cardY
        self.p_y= (self.m_y + m_y_prior)/np.sum(self.m_y+ m_y_prior)

        if w_mean0>0:
            # Prior parameters
            prior_mean = np.ones(self.n)* mean0 # Prior mean matrix
            prior_cov = np.eye(self.n)*cov0  # Prior covariance matrix


        self.mean_y= np.zeros((self.cardY,self.n))
        self.cov_y= np.zeros((self.cardY,self.n,self.n))
        if w_mean0== 0:
            for c in np.arange(self.cardY):
                self.mean_y[c,:]= self.s1_y[c,:]/self.m_y[c]
                self.cov_y[c,:,:]= self.s2_y[c,:,:]/self.m_y[c] - np.outer(self.mean_y[c,:],self.mean_y[c,:])
        elif w_mean0>0:
            for c in np.arange(self.cardY):
                mu= self.s1_y[c,:]/self.m_y[c]
                cov= self.s2_y[c,:,:]/self.m_y[c] - np.outer(self.mean_y[c,:],self.mean_y[c,:])

                self.mean_y[c,:]= (mu * self.m_y[c] + w_mean0 * prior_mean)/(self.m_y[c] + w_mean0)
                # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bayesian_inference
                # Taking the modes for p(mu| Sigma,X) and for p(Sigma|X) using the conjugate priors (normal for the mean
                # and inverse Wishart for the covariance)
                #self.cov_y[c, :, :] = (prior_cov + self.m_y[c] * cov + (
                #            self.m_y[c] * w_cov0 / (self.m_y[c] + w_cov0)) * (
                #            mu - prior_mean) @ (mu - prior_mean).transpose()) / (self.m_y[c] + w_cov0 - self.n - 1)

                # Looking at "Introduction to Bayesian Data Imputation" (Holt,Nguyen) Eq.5 and the next equation regarding the
                # posterior mean there are errors in Wikipedia.

                # The posterior distribution for the covariance given the data is asumming p(Sigma) is an inverse
                # Wishart with parameters S_0^-1, nu_0 is a Wishart with parameters [S_0 + S]^-1, nu_o + n where
                # S is n*cov and n is the number of data points.
                # According to (Holt,Nguyen) E[Sigma|X,mu)= (S_0 + S)/(nu_0-dim-1 + n)
                # After a reparametrization, geting one iteration of Gibbs sampling starting at mu and setting
                # prior_cov= S_0/w_cov with w_cov= nu_0 - dim -1 we get the more intuidive
                self.cov_y[c, :, :]= (self.m_y[c]* cov + w_cov0 * prior_cov)/(self.m_y[c] + w_cov0)

    def fit(self,X,Y, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0):
        self.m_y, self.s1_y, self.s2_y= self.getStats(X,Y)
        self.statsToParams(ess= ess, mean0= mean0, w_mean0=w_mean0, cov0= cov0, w_cov0= cov0)

    def getClassProbs(self, X):
        m= X.shape[0]
        pY= np.zeros((m,self.cardY))
        for c in np.arange(self.cardY):
            mvn = multivariate_normal(mean=self.mean_y[c,:], cov=self.cov_y[c,:,:], allow_singular= True)
            pY[:,c]= mvn.pdf(X)*self.p_y[c]

        pY/= np.sum(pY, axis= 1, keepdims= True)
        return pY

    def riskDesc(self, X, Y, lr= 0.1, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0, correct_forbidden_stats= False):

        oh = np.eye(self.cardY)[Y]
        pY = self.getClassProbs(X)

        dif = pY - oh

        # compute the change on the statistics
        d_m_y, d_s1_y, d_s2_y = self.getStats(X, dif)
        self.m_y -= lr * d_m_y
        self.s1_y -= lr * d_s1_y
        self.s2_y -= lr * d_s2_y

        if correct_forbidden_stats:
            if np.any(self.m_y < 0):
                self.m_y -= lr * d_m_y

            #deberia hacerse para todos los valores de la clase o para ninguno: concatenar las features y despues dorregir todos los condicionadaos de el conjunto de features
            for y in range(self.cardY):
                feats= np.where((np.diag(self.s2_y[y,:,:]- self.s1_y[y,:])**2/self.m_y[y])<= 0)[0]
                if len(feats)>0:
                    None
                self.s2_y[y,:,:][np.ix_(feats,feats)]+= lr * d_s2_y[y,:,:][np.ix_(feats,feats)]
                self.s1_y[y][feats]+= lr * d_s1_y[y][feats]

        # update the parameters
        self.statsToParams(ess, mean0, w_mean0, cov0, w_cov0)

    def predict(self, X):
        pY= self.getClassProbs(X)
        return np.argmax(pY, axis= 1)
