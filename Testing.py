import Utils as utl
import numpy as np
from Models import NaiveBayesDisc
from Models import QDA
from  Models import LinearClassifier
from sklearn.preprocessing import KBinsDiscretizer
from Utils import preprocess_data

class MLR:
    '''
    Multinomial logistic regression

    Code taken from internet with sanity check purposes

    The implementation of the gradient descent should be similar to the implementation of gradient descent with
    parameters initialized to zero for LogisticReg classifier with canonical=False
    '''

    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.iters = n_iters
        self.W = None

    def fit(self, X, y, mu= 0.0, paramStart= False):
        '''
        :param X: training data of the features
        :param y: training data of the class
        :param mu: L2 penalization, by default no penalization
        :return:
        '''
        ones = np.ones(X.shape[0])
        # add a first feature corresponding to the free parameter that only depends on the class
        features = np.c_[ones, X]
        onehot_encoder = OneHotEncoder(sparse=False)
        y_encode = onehot_encoder.fit_transform(y.reshape(-1, 1))
        # initialize the parameters of the model to 0
        if paramStart:
            # todo: inicializacion parametrica
            self.W = np.zeros((features.shape[1], y_encode.shape[1]))
        else:
            self.W = np.zeros((features.shape[1], y_encode.shape[1]))
        samples = X.shape[0]

        logloss= list()

        for i in range(self.iters):
            # Originalmente esta pensado para que se clasifique con -Theta_y*T(x).
            # Lo pongo en terminos de Theta_y*T(x) para que se parezca mas a la forma canonica exponencial
            #Z = -features @ self.W

            Z = features @ self.W
            prob_y = softmax(Z, axis=1)

            # compute the log loss
            logloss.append(np.average(- np.log(prob_y[np.arange(samples), y])))

            #Calculo del gradiente, para (x,y) and all y': p(y'|x) - [y=y']
            #error = y_encode - prob_y# negative params
            neg_error = prob_y - y_encode
            dW = 1 / samples * (features.T @ neg_error) + 2 * mu * self.W

            # Descenso del gradiente con el learning rate lr
            self.W -= self.lr * dW

        return np.array(logloss)

    def getClassProb(self, X):
        ones = np.ones(X.shape[0])
        features = np.c_[ones, X]
        #Z = -features @ self.W
        Z = features @ self.W
        y = softmax(Z, axis=1)
        return y
    def predict(self, X):
        return np.argmax(self.getClassProb(X), axis=1)


def pruebas_discriminant(epochs= 128, lr= 0.1):
    '''
    pruebas con logistic regression using gradient descent (GD) VS using ERD y las BDs


    '''

    dataNames = ['ecoli', 'optdigits', 'iris', 'adult', 'satellite', 'vehicle', 'segment', 'redwine', 'letterrecog',
                 'forestcov', 'haberman', 'mammographic', 'indian_liver', 'heart', 'sonar', 'svmguide3',
                 'liver_disorder', 'german_numer']

    for dataName in dataNames:

        try:
            X, origY = eval("utl.load_"+ dataName + '(return_X_y=True)')
            Y= utl.normalizeLabels(origY)
            esz = 0.0
            cardY= np.unique(Y).shape[0]

            m= X.shape[0]
            # Randomize and selec 1000 instances
            perm= np.random.permutation(m)[:np.min([1000,m])]
            m= np.min([1000,m])
            X= X[perm,:]
            Y= Y[perm]

            # Remove constant columns and normalize
            X = utl.preprocess_data(X,Y)
            m, n = X.shape

            # Initialize KBinsDiscretizer with strategy='quantile' for equal frequency binning
            # n_bins specifies the number of bins
            discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')

            # Fit and transform the data
            X_discretized = discretizer.fit_transform(X).astype(int)
            unique_counts = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X_discretized)
            if np.any(unique_counts < 2):
                X_discretized= X_discretized[:,unique_counts==2]
                n= X_discretized.shape[1]

            print(dataName)
            print("########")
            print("(m,n)=" + str((m,n)))
            print("card Y= " + str(cardY))
            print("proportions: "+ str(np.unique(Y, return_counts= True)[1]))
            # Set printing options
            np.set_printoptions(precision=2, suppress=True)
            print("mean")
            print(np.average(X, axis= 0))
            np.set_printoptions(precision=2, suppress=True)
            print("var")
            print(np.var(X, axis= 0))

            models = ["QDA"]#, "LR-MS GD", "LR RD-MS", "NB-ML RD", "NB-ML GD", "NB-unif RD", "NB-unif GD"]
            for model in models:

                if model in ["QDA"]:
                    try:
                        print(model)
                        h= QDA(cardY,n)
                        h.fit(X,Y)
                        print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " " + str(
                            np.average(Y != h.predict(X))) + " " + str(np.average(1 - h.getClassProbs(X)[np.arange(m), Y])))
                        for i in range(1, epochs):
                            h.riskDesc(X, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i + 1)):
                                print(str(i) + ": " + str(
                                    np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " " + str(
                                    np.average(Y != h.predict(X))) + " " + str(
                                    np.average(1 - h.getClassProbs(X)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)
                if model == "NB-ML":
                    try:
                        print(model)
                        h= NaiveBayesDisc(cardY, n)
                        h.fit(X_discretized,Y)
                        print(str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "NB-ML RD":
                    try:
                        print(model)
                        h = NaiveBayesDisc(cardY, n)
                        h.fit(X_discretized, Y)
                        print(str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(
                            np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                        for i in range(1, epochs):
                            h.riskDesc(X_discretized, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i + 1)):
                                print(str(i) + ": " + str(
                                    np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(
                                    np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "NB-ML GD":
                    try:
                        print(model)
                        h = NaiveBayesDisc(cardY, n)
                        h.fit(X_discretized, Y)
                        print(str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(
                            np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                        for i in range(1, epochs):
                            h.gradDesc(X_discretized, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i + 1)):
                                print(str(i) + ": " + str(
                                    np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(
                                    np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "NB-unif RD":
                    try:
                        print(model)
                        h = NaiveBayesDisc(cardY, n)
                        h.initialize(m)
                        print(str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                        for i in range(1, epochs):
                            h.riskDesc(X_discretized, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i + 1)):
                                print(str(i) + ": " + str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "NB-unif GD":
                    try:
                        print(model)
                        h = NaiveBayesDisc(cardY, n)
                        h.initialize(m)
                        print(str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                        for i in range(1, epochs):
                            h.gradDesc(X_discretized, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i + 1)):
                                print(str(i) + ": " + str(np.average(- np.log(h.getClassProbs(X_discretized)[np.arange(m), Y]))) + " " + str(np.average(Y != h.predict(X_discretized)))+ " " + str(np.average(1- h.getClassProbs(X_discretized)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "LR RD-MS":
                    try:
                        print(model)
                        h= LinearClassifier(cardY,n)
                        h.fit(X,Y)
                        print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " "+ str(np.average(Y!=h.predict(X)))+ " " + str(np.average(1- h.getClassProbs(X)[np.arange(m), Y])))
                        for i in range(1,epochs):
                            h.riskDesc(X, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i+1)):
                                print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " "+ str(np.average(Y!=h.predict(X)))+ " " + str(np.average(1- h.getClassProbs(X)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)

                if model == "LR-MS GD":
                    try:
                        print(model)
                        h = LogReg(n, cardY, canonical= False)
                        h.minimumSquare(X,Y)
                        print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " "+ str(np.average(Y!=h.predict(X)))+ " " + str(np.average(1- h.getClassProbs(X)[np.arange(m), Y])))
                        for i in range(1,epochs):
                            h.gradDesc(X, Y, lr)
                            if int(np.log2(i)) < int(np.log2(i+1)):
                                print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))) + " "+ str(np.average(Y!=h.predict(X)))+ " " + str(np.average(1- h.getClassProbs(X)[np.arange(m), Y])))
                    except Exception as e:
                        # Handling the exception by printing its description
                        print("An exception occurred:", e)
        except Exception as e:
            # Handling the exception by printing its description
            print("Exception in data " + dataName, e)


def pruebas_logisticRegression(lr= 0.1):
    '''
    pruebas con logistic regression using gradient descent (GD) VS using ERD y las BDs


    '''

    #divergen
    dataNames = ['iris', 'segment']
    #empieza peor
    #dataNames = ['sonar', 'satellite']
    #constante
    #dataNames = ['indian_liver']
    #cascan
    #dataNames = ['haberman', 'mammographic', 'ecoli']
    # las que van bien
    #dataNames = ['heart', 'svmguide3', 'liver_disorder', 'german_numer', 'adult', 'optdigits', 'vehicle', 'redwine', 'letterrecog', 'forestcov']
    for dataName in dataNames:

        X, origY = eval("utl.load_"+ dataName + '(return_X_y=True)')
        Y= utl.normalizeLabels(origY)
        esz = 0.0
        cardY= np.unique(Y).shape[0]

        # Remove constant columns and normalize
        X = preprocess_data(X)
        m,n= X.shape
        perm= np.random.permutation(m)
        X= X[perm,:]
        Y= Y[perm]

        print(dataName)
        print("########")
        print("(m,n)=" + str((m,n)))
        # Set printing options
        np.set_printoptions(precision=2, suppress=True)
        print("mean")
        print(np.average(X, axis= 0))
        np.set_printoptions(precision=2, suppress=True)
        print("var")
        print(np.var(X, axis= 0))

        maxiter= 100

        try:
            print("grad desc")
            h = LogReg(n, cardY, canonical= False)
            h.initialization()
            print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10): h.gradDesc(X, Y, lr); print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10,maxiter): h.gradDesc(X, Y, lr)
            print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
        except Exception as e:
            # Handling the exception by printing its description
            print("iteration " + str(i), e)

        try:
            print("canonical grad desc")
            h = LogReg(n, cardY, canonical= True)
            h.fit(X,Y)
            print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10): h.gradDesc(X, Y, lr); print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10,maxiter): h.gradDesc(X, Y, lr)
            print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
        except Exception as e:
            # Handling the exception by printing its description
            print("iteration " + str(i), e)

        try:
            print("risk descent")
            h = LogReg(n, cardY, canonical= True)
            h.fit(X,Y)
            loss= np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))
            print(str(loss))
            for i in range(10):
                loss = np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))
                h.riskDesc(X, Y, lr, loss); print(str(i)+": " + str(loss))
            for i in range(10,maxiter):
                loss = np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))
                h.riskDesc(X, Y, lr, loss)
                if (i % 10)==0:  print(str(i)+": " + str(loss))
        except Exception as e:
            # Handling the exception by printing its description
            print("iteration " + str(i), e)
        try:
            print("stoc risk descent")
            h = LogReg(n, cardY, canonical= True)
            h.fit(X,Y)
            print(str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10): h.stotRiskDesc(X, Y, lr*0.1, stc= False); print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
            for i in range(10,maxiter):
                h.stotRiskDesc(X, Y, lr*0.1, stc= False)
                if i%10==0:  print(str(i)+": " + str(np.average(- np.log(h.getClassProbs(X)[np.arange(m), Y]))))
        except Exception as e:
            # Handling the exception by printing its description
            print("iteration " + str(i), e)

dataNames= ["blood_transfusion", "climate_model", "diabetes", "ionosphere", 'magic', 'pulsar', 'QSAR', 'splice', 'glass',
            'yearbook_resnet18', 'catsvsdogs_resnet18', 'mnist_resnet18']

def load_data():

    for dataName in dataNames:
        X, Y = eval("utl.load_" + dataName + '(return_X_y=True)')
        X, Y= preprocess_data(X, Y)
        cardY = np.unique(Y).shape[0]
        m, n = X.shape

        print(dataName)
        print("########")
        print("(m,n)=" + str((m, n)))
        print("card Y= " + str(cardY))
        print("proportions: " + str(np.unique(Y, return_counts=True)[1]))
        # Set printing options
        np.set_printoptions(precision=2, suppress=True)
        print("mean")
        print(np.average(X, axis=0))
        np.set_printoptions(precision=2, suppress=True)
        print("var")
        print(np.var(X, axis=0))

if __name__ == '__main__':
    load_data()