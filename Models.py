import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax
import scipy.linalg as la

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

def closest_non_singular_matrix(A, epsilon=1e-6):
    """
    Obtain the closest non-singular matrix to a given singular matrix by perturbing its singular values.

    Parameters:
        A (numpy.ndarray): Singular matrix.
        epsilon (float): Small positive value for replacing zero singular values.

    Returns:
        numpy.ndarray: Closest non-singular matrix.
    """
    # Compute singular value decomposition
    U, Sigma, Vt = np.linalg.svd(A, hermitian= True)

    # Replace zero singular values with a small positive value
    Sigma[Sigma < epsilon] = epsilon

    # Reconstruct matrix with modified singular values
    A_prime = U @ np.diag(Sigma) @ Vt

    return A_prime

def make_positive_definite_eig(matrix, epsilon=1e-6):
    """
    Ensure the matrix is positive definite by adjusting its eigenvalues.

    Parameters:
    matrix (np.ndarray): The input symmetric matrix.
    epsilon (float): The small value to add to the eigenvalues if they are non-positive.

    Returns:
    np.ndarray: The adjusted positive definite matrix.
    """
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix is not symmetric")

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Adjust the eigenvalues to be positive
    adjusted_eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct the matrix
    positive_definite_matrix = (eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T)

    return positive_definite_matrix

def log_multivariate_gaussian(X, mean, cov):
    """
    Compute the log-pdf of multivariate Gaussian distribution for multiple instances.

    Parameters:
        X (numpy.ndarray): Input matrix where each row represents an instance.
        mean (numpy.ndarray): Mean vector of the Gaussian distribution.
        cov (numpy.ndarray): Covariance matrix of the Gaussian distribution.

    Returns:
        numpy.ndarray: Log-pdf of the multivariate Gaussian distribution for each instance.
    """
    # Compute the log determinant of the covariance matrix
    log_det_cov = np.linalg.slogdet(cov)[1]

    # Compute the Mahalanobis distance for each instance (use no.linalg.pinv, the pseudoinverse)
    mahalanobis = np.sum(np.dot((X - mean), np.linalg.pinv(cov)) * (X - mean), axis=1)

    # Compute the log-pdf for each instance
    log_pdf = -0.5 * (log_det_cov + mahalanobis + len(mean) * np.log(2 * np.pi))
    return log_pdf

class NaiveBayesDisc:
    '''
    Naive Bayes classifiers for categorical input features (NB)
    '''
    def __init__(self, cardY, card, n):
        '''
        Constructor for NB

        Args:
            cardY: cardinality of the class variable
            card: cardinality of the discrete input features
            n: number of features
        '''
        self.cardY = cardY
        self.num_features = n
        self.card= card
        self.class_counts = np.zeros(cardY)
        self.feature_counts = np.zeros((n, card, cardY))
        self.class_probs = np.zeros(cardY)
        self.cond_probs = np.zeros((n, card, cardY))

    def getStats(self, X, Y):
        '''
        The statistics mapping for NB
        Args:
            X: training unlabeled instances
            Y: training class labels

        Returns:
            The statistics obtained from training data for NB: counting statistics
        '''
        m,n= X.shape
        class_counts = np.zeros(self.cardY)
        feature_counts = np.zeros((n, self.card, self.cardY))

        if Y.ndim == 1:
            # Compute class counts
            class_counts += np.bincount(Y, minlength=self.cardY)

            # Compute feature counts
            for j in range(self.num_features):
                feature_counts[j, :, :]= np.bincount(X[:,j]*self.cardY+ Y, minlength= self.card* self.cardY ).reshape((self.card,self.cardY))

        else:
            # Compute class counts
            class_counts += np.sum(Y,axis= 0)

            # Compute input feature counts
            for j in range(self.num_features):
                for x in range(self.card):
                    feature_counts[j, x, :] = np.sum(Y[X[:, j]==x,:], axis= 0)


        return class_counts, feature_counts

    def statsToParams(self, ess= 0):
        '''
        Parameter mapping for NB that gets and stores the parameters from the statistics.

        The statistics are obtained from the attributes: class_counts and feature_counts-

        Args:
            ess: when zero corresponds to maximum likelihood parameters. When positive corresponds to maximum a
            posteriori parameters with a uniform dirichlet prior with ess equivalent sample size
        '''

        if ess>0:
            # Priors for the maximum a posteriori estimates
            feature_counts_prior = np.ones((self.num_features, self.card, self.cardY)) * ess / (self.card * self.cardY)
            class_counts_prior = np.ones(self.cardY) * ess / self.cardY

        # Compute class probabilities
        if ess==0:
            self.class_probs = self.class_counts / np.sum(self.class_counts)
        elif ess>0:
            self.class_probs = (self.class_counts + class_counts_prior) / np.sum(self.class_counts + class_counts_prior)

        # Compute conditional probabilities
        for j in range(self.num_features):
            for f in range(self.card):
                for c in range(self.cardY):
                    if ess== 0:
                        self.cond_probs[j, f, c] = self.feature_counts[j, f, c] / self.class_counts[c]
                    elif ess>0:
                        self.cond_probs[j, f, c] = (self.feature_counts[j, f, c] + feature_counts_prior[j, f, c]) / (
                                    self.class_counts[c] + class_counts_prior[c])

    def fit(self, X, Y, ess= 0):
        '''
        The closed-form learning algorithm for NB.

        When ess=0 learn the maximum likelihood parameters. When ess>0 learn the maximum a posteriori parameters
        with a Dirichlet prior distribution with uniform hyperparameters and an equivalent sample size of ess

        Args:
            X: training unlabeled instances
            Y: training class labels
            ess: equivalent sample size of the Dirichlet prior

        '''
        self.class_counts,self.feature_counts= self.getStats(X, Y)
        self.statsToParams(ess)

    def riskDesc(self, X, Y, lr= 0.1, ess= 0, correct_forbidden_stats= True):
        '''
        Risk-based calibration learning algorithm for NB. It is based on the maximum likelihood/maximum a posteriori
        closed form learning algorithm ess=0/ess>0

        Args:
            X: training unlabeled instances
            Y: training class labels
            lr: learning rate
            ess: equivalent sample size of the Dirichlet prior
            correct_forbidden_stats: when true (default) it does not update the statistics with negative statistics

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
                self.class_counts += lr * d_class_counts

            first, second= np.where((self.feature_counts < 0).any(axis=(1)))
            indices= list(zip(first, second))
            for j,y in indices:
                self.feature_counts[j, :, y] += lr * d_feature_counts[j, :, y]

        # update the parameters from the parameters
        self.statsToParams(ess)


    def gradDesc(self, X, Y, lr= 0.1, opt_normalization= 0):
        '''
        Gradient descent of the average negative log loss in training data for NB. The gradient is computed for the
        parameters (probabilities) in their logarithm form to avoid negative probabilities. Once the gradient descent
        is computed the parameters are projected into the simplex to guarantee that represent probabilities

        :param X: unlabeled instances
        :param Y: labels
        :param lr: learning rate
        :opt_normalization: normalization of the parameters to reflect probability tables.
            O (default): project into the simplex
            1: project using softmax
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

        if opt_normalization== 0:
            # Normalize the probability table by projecting into the simplex
            self.class_probs = project_onto_simplex(np.exp(alpha_y))
        else:
            # Normalization using softmax
            self.class_probs= softmax(alpha_y)

        d_beta_y= np.zeros(self.card)
        for j in np.arange(self.num_features):
            for c in np.arange(self.cardY):
                # For each probability table: feature k and class c
                for x in np.arange(self.card):
                    d_beta_y[x]= np.sum(dif[X[:,j]== x,c])/m

                beta_y = np.log(self.cond_probs[j,:,c])
                beta_y -= lr * d_beta_y

                if opt_normalization== 0:
                    # Normalize each probability table by projecting into the simplex
                    self.cond_probs[j, :, c] = project_onto_simplex(np.exp(beta_y))
                else:
                    #Normalization using softmax
                    self.cond_probs[j, :, c]= softmax(beta_y)

    def DFE(self, X, Y, lr= 0.1, ess= 0):
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

        m= X.shape[0]
        pY = self.getClassProbs(X)
        E= np.zeros((m,self.cardY))
        E[np.arange(m), Y] = 1- pY[np.arange(m), Y]

        # compute the change on the statistics
        d_class_counts, d_feature_counts = self.getStats(X, E)

        self.class_counts += lr * d_class_counts
        self.feature_counts += lr * d_feature_counts

        # update the parameters from the parameters
        self.statsToParams(ess)


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
                # According to (Holt,Nguyen, p.17, first equation) E[Sigma|X,mu)= (S_0 + S)/(nu_0-dim-1 + n), where
                # S_0 and Sare the prior and sample sum sum_i=1^n (x_i - theta)^T(x_i - theta).
                # After a reparametrization, geting one iteration of Gibbs sampling starting at mu and setting
                # prior_cov= S_0/w_cov with w_cov= nu_0 - dim -1 we get the more intuidive
                self.cov_y[c, :, :]= (self.m_y[c]* cov + w_cov0 * prior_cov)/(self.m_y[c] + w_cov0)

    def fit(self,X,Y, ess= 0, mean0= 0, w_mean0=0, cov0= 1, w_cov0= 0):
        self.m_y, self.s1_y, self.s2_y= self.getStats(X,Y)
        self.statsToParams(ess= ess, mean0= mean0, w_mean0=w_mean0, cov0= cov0, w_cov0= cov0)

    def getClassProbs(self, X):

        m= X.shape[0]
        pY= np.zeros((m,self.cardY))
        #aux_pY= np.zeros((m,self.cardY))
        for c in np.arange(self.cardY):
            log_mvn= log_multivariate_gaussian(X= X, mean=self.mean_y[c,:], cov=self.cov_y[c,:,:])
            # avoid infinity
            log_mvn[np.where(np.isinf(log_mvn))] = -10 ** 12
            pY[:, c] = log_mvn + np.log(self.p_y[c])
            #mvn = multivariate_normal(mean=self.mean_y[c,:], cov=self.cov_y[c,:,:], allow_singular= True)
            #aux_pY[:,c]= mvn.pdf(X)*self.p_y[c]

        #aux_pY/= np.sum(aux_pY, axis= 1, keepdims= True)
        pY= softmax(pY, axis=1)

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
            # If any statistic is not valid undo the changes

            # class marginal
            if np.any(self.m_y < 10**-3):
                self.m_y += lr * d_m_y
            # class conditional covariance

            epsilon = 10 ** -6
            for y in range(self.cardY):
                if np.any(np.diag(self.s2_y[y]) <= 0):
                    self.s1_y += lr * d_s1_y
                    self.s2_y += lr * d_s2_y
                '''
                var= np.diag(self.s2_y[y])/self.m_y[y] - (self.s1_y[y]/self.m_y[y])**2
                lowest= np.argsort(var)[:np.min([15,self.n])]
                cov= self.s2_y[y][np.ix_(lowest,lowest)]/self.m_y[y] - np.outer(self.s1_y[y][lowest],self.s1_y[y][lowest])/self.m_y[y]**2
                if (np.linalg.det(cov) < epsilon) or np.any(var < 0):
                    self.s2_y[y]+= lr * d_s2_y[y]
                    self.s1_y[y]+= lr * d_s1_y[y]
                '''

        # update the parameters
        self.statsToParams(ess, mean0, w_mean0, cov0, w_cov0)

    def gradDesc(self, X, Y, lr= 0.1, psd_cov= True):

        m,n= X.shape


        # one-hot encoding of Y
        oh = np.zeros((m, self.cardY))
        oh[np.arange(m), Y] = 1
        pY= self.getClassProbs(X)

        dif= pY - oh
        # avoid class conditional probabilities with NaNs.
        nan_rows= np.isnan(dif).any(axis= 1)
        dif= dif[~nan_rows, :]
        X= X[~nan_rows, :]
        m= dif.shape[0]
        sum= np.sum(dif, axis= 0)


        # Change the parameters to the exponential form
        # log p(y)
        nu_0= np.log(self.p_y)
        #cov^-1 * mu
        nu_1= np.zeros_like(self.mean_y)
        #-1/2 * cov^-1
        nu_2= np.zeros_like(self.cov_y)
        nu_2_inv= np.zeros_like(self.cov_y)
        for c in range(self.cardY):
            inv_cov= np.linalg.pinv(self.cov_y[c])
            nu_1[c]= np.dot(inv_cov, self.mean_y[c])
            nu_2[c]= -0.5 * inv_cov
            nu_2_inv[c]= -2 * self.cov_y[c]


        # Compute the gradients of the parameters in the exponential form
        d_nu_0= sum/m

        #d R(nu)/d nu_1|d=  1/m sum_{x,y} [y==d](x + 1/2 nu_2|d^-1 \nu_1|d

        d_nu_1= np.einsum('ai,aj->ij', dif, X)/m
        for c in np.arange(self.cardY):
            d_nu_1[c]+= 0.5* sum[c] * np.matmul(nu_2_inv[c], nu_1[c])/m


        #d R(nu)/d nu_2|d= 1/m sum_{x,y} (h(d|x) - [d==y])(x·x^t - 1/4 · nu_2|d^-1 · nu_1,d · nu_1|d^t · nu_2|d^-1 + 1/2 tr(nu_2|d^-1)
        d_nu_2= np.zeros((self.cardY, self.n, self.n))
        for c in np.arange(self.cardY):
            d_nu_2[c]= np.einsum('a,ai,aj->ij', dif[:,c], X, X)/m
            d_nu_2[c]+= -0.25 * sum[c] * np.dot(np.dot(nu_2_inv[c], np.outer(nu_1[c],nu_1[c])), nu_2_inv[c])/m
            d_nu_2[c]+= 0.5* sum[c] * np.trace(nu_2_inv[c])/m

        # Apply the gradient
        nu_0 -= lr * d_nu_0
        nu_1 -= lr * d_nu_1
        nu_2 -= lr * d_nu_2

        # Transform the parameters into the standard form and transorm to get valid param values: probs, means and covariances
        # Get probabilities using softmax
        # p_y= exp{nu_0}
        self.p_y = softmax(nu_0)
        epsilon= 10**-3
        for c in range(self.cardY):
            #mu(y)= -1/2 nu_2(y)^-1 nu_1(y)
            self.mean_y[c]= -0.5* np.linalg.inv(nu_2[c]).dot(nu_1[c])

            #cov(y)= -1/2 nu_2(y)^-1
            self.cov_y[c]= -0.5* np.linalg.inv(nu_2[c])

            if psd_cov:
                if self.n< 50:
                    # Get the closes covarianze matrix using eignevalue decomposition
                    self.cov_y[c] = closest_non_singular_matrix(self.cov_y[c], epsilon= epsilon)
                else:
                    np.fill_diagonal(self.cov_y[c], np.maximum(np.diagonal(self.cov_y[c]), epsilon))




    def predict(self, X):
        pY= self.getClassProbs(X)
        return np.argmax(pY, axis= 1)
