# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt

#from matplotlib import pyplot as plt
#from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import sklearn.svm

from Models import LinearClassifier, NaiveBayesDisc, QDA
from LogReg import LogReg
#from LogRegOld import LogRegOld
from Utils import preprocess_data
import Utils as utl
import os

dataNames = ['glass', 'optdigits' , 'splice',  'QSAR', 'letterrecog', 'vehicle', 'satellite', 'sonar','adult','redwine',
             'svmguide3', 'ecoli', 'german_numer', 'haberman', 'heart', 'indian_liver', 'iris', 'liver_disorder', 'mammographic',
              "blood_transfusion", "climate_model", "diabetes", "ionosphere", 'magic', 'pulsar']#, 'glass' falla log reg]#, 'thyroid']

dataNames2 = ["mnist", "catsvsdogs", "cifar10", "fashion_mnist", "yearbook"]

def experiment_embedded_data_LR(lr=0.1, numIter=256, seed=0):
    """
    Experiment to analyze curves of convergence with the data based on ResNet18 embeddings for Logistic Regression
    Args:
        lr:
        numIter:
        seed:

    Returns:

    """

    # np.random.seed(seed)
    algorithms = ["GD", "RD"]
    types = ["ML", "MS"]
    dataNames = ["mnist", "catsvsdogs", "cifar10", "fashion_mnist", "yearbook"]
    #dataNames = ["catsvsdogs"]
    res = []

    cont_plot = 0
    for d in range(len(dataNames)):
        # For each dataset
        X, Y = eval("utl.load_" + dataNames[d] + '_features_resnet18(with_info=False, split=False)')
        X, Y = preprocess_data(X, Y)
        m, n = X.shape
        cardY = np.unique(Y).shape[0]
        print(m)
        print(n)
        print(X[1,1])

        for type in types:
            for algorithm in algorithms:
                if algorithm == "GD" and type == "ML":
                    h = LogReg(n, cardY, canonical=False)
                    h.fit(X, Y)
                elif algorithm == "GD" and type == "MS":
                    h = LogReg(n, cardY, canonical=False)
                    h .minimumSquare(X, Y)
                elif algorithm == "RD" and type == "ML":
                    h = LogReg(n, cardY, canonical=True)
                    h.fit(X, Y)
                elif algorithm == "RD" and type == "MS":
                    # TODO: integrarlo en LogReg. fit y riskDesc deben tener type
                    # TODO: lo de canonical solo tiene utilidad con riskDesc + ML
                    h = LinearClassifier(n, cardY)
                    h.fit(X, Y)
                pY = h.getClassProbs(X)
                prevError = np.inf
                actError = np.average(1 - pY[np.arange(m), Y])
                try:
                    iter = 1
                    res_it = []
                    res_it.append(actError)
                    print(f"{algorithm}")
                    # print(f"iter actError     (prevError-actError)  (prevError-actError)< 0.001")
                    while iter < numIter:
                        # print(f"{iter} {actError} {prevError - actError} {prevError - actError < 0.001}")
                        iter += 1
                        prevError = actError
                        if algorithm == "RD":
                            h.riskDesc(X, Y, lr)
                        else:
                            h.gradDesc(X, Y, lr)
                        pY = h.getClassProbs(X)
                        actError = np.average(1 - pY[np.arange(m), Y])
                        if prevError - actError < 0.0001:
                            break
                        res_it.append(actError)
                        # print(f"{iter} {actError} {prevError - actError} {prevError - actError < 0.001}")
                    res.append([algorithm + " - " + type, res_it])
                except Exception as e:
                    # Handling the exception by printing its description
                    print(f"Exception at iter {iter}:\n {e}")

            print(res)
            plt.figure(figsize=(10, 6))
            plt.plot(res[cont_plot][1])
            plt.plot(res[cont_plot+1][1])
            plt.title('Error vs. Iteration in ' + dataNames[d] + " lr: " + str(lr))
            plt.xlabel('Number of Iterations')
            plt.ylabel('Error')
            plt.legend([res[cont_plot][0], res[cont_plot+1][0]], loc="lower left")
            plt.savefig("./Results/LogReg_" + dataNames[d] + type + ".png")
            plt.show()
            cont_plot = cont_plot + 2


def experiment_embedded_data_NB(lr=0.1, numIter=128, seed=0):
    """
    Experiment to analyze curves of convergence with the data based on ResNet18 embeddings for discrete Naive Bayes
    Args:
        lr:
        numIter:
        seed:

    Returns:

    """

    # np.random.seed(seed)
    algorithms = ["GD", "RD"]
    types = ["ML", "MAP"]
    dataNames = ["mnist", "catsvsdogs", "cifar10", "fashion_mnist", "yearbook"]
    #dataNames = ["mnist"]
    res = []

    cont_plot = 0
    for d in range(len(dataNames)):
        # For each dataset
        X, Y = eval("utl.load_" + dataNames[d] + '_features_resnet18(with_info=False, split=False)')
        #X, Y = eval("utl.load_" + dataNames[d] + '(return_X_y=True)')
        card = 2
        X, Y = preprocess_data(X, Y, n_bins=card)
        m, n = X.shape
        cardY = np.unique(Y).shape[0]
        print(m)
        print(n)
        ess=cardY
        for type in types:
            for alg in algorithms:
                h= NaiveBayesDisc(cardY, card, n)

                if type== "ML":
                    h.fit(X, Y)
                elif type== "MAP":
                    h.fit(X, Y, ess= ess)

                h.fit(X, Y)
                pY = h.getClassProbs(X)
                prevError = np.inf
                actError = np.average(1 - pY[np.arange(m), Y])
                actError1 = np.average(Y != np.argmax(pY, axis=1))
                print(actError1)
                print(actError)
                try:
                    iter = 1
                    res_it = []
                    res_it.append(actError)
                    print(f"{alg}")
                    # print(f"iter actError     (prevError-actError)  (prevError-actError)< 0.001")
                    while iter < numIter:
                        print(iter)
                        iter += 1
                        prevError = actError

                        if alg == "RD":
                            if type == "ML":
                                h.riskDesc(X, Y, lr)
                            elif type == "MAP":
                                h.riskDesc(X, Y, lr, ess=ess)
                        elif alg == "GD":
                            h.gradDesc(X, Y, lr)
                        pY = h.getClassProbs(X)

                        actError = np.average(1 - pY[np.arange(m), Y])
                        actError1 = np.average(Y != np.argmax(pY, axis=1))
                        print(actError1)
                        print(actError)
                        if prevError - actError < 0.0001:
                            break
                        res_it.append(actError)
                        # print(f"{iter} {actError} {prevError - actError} {prevError - actError < 0.001}")
                    res.append([alg + " - " + type, res_it])
                except Exception as e:
                    # Handling the exception by printing its description
                    print(f"Exception at iter {iter}:\n {e}")
                    res.append([alg + " - " + type, res_it])

            print(res)
            plt.figure(figsize=(10, 6))
            plt.plot(res[cont_plot][1])
            plt.plot(res[cont_plot+1][1])
            plt.title('Error vs. Iteration in ' + dataNames[d] + " lr: " + str(lr))
            plt.xlabel('Number of Iterations')
            plt.ylabel('Error')
            plt.legend([res[cont_plot][0], res[cont_plot+1][0]], loc="lower left")
            plt.savefig("./Results/NB2_" + dataNames[d] + type + ".png")
            #plt.show()
            cont_plot = cont_plot + 2


def experiment_embedded_data_QDA(lr=0.1, numIter=128, seed=0):
    """
    Experiment to analyze curves of convergence with the data based on ResNet18 embeddings for Logistic Regression
    Args:
        lr:
        numIter:
        seed:

    Returns:

    """

    # np.random.seed(seed)
    classif = "QDA"
    algorithm= "RD"
    dataNames = ["mnist", "catsvsdogs", "cifar10", "fashion_mnist", "yearbook"]
    #dataNames = ["cifar10"]
    res = []

    cont_plot = 0
    for d in range(len(dataNames)):
        try:
            # For each dataset
            X, Y = eval("utl.load_" + dataNames[d] + '_features_resnet18(with_info=False, split=False)')
            X, Y = preprocess_data(X, Y)
            m, n = X.shape
            cardY = np.unique(Y).shape[0]
            print(m)
            print(n)
            print(X[1, 1])

            h = QDA(cardY, n)
            h.fit(X, Y)

            #  pY parece que pone 1 en el Y real
            pY = h.getClassProbs(X)
            prevError = np.inf
            actError = np.average(1 - pY[np.arange(m), Y])

            iter = 1
            res_it = []
            res_it.append(actError)
            print(f"{algorithm}")
            # print(f"iter actError     (prevError-actError)  (prevError-actError)< 0.001")
            while iter < numIter:
                if prevError - actError < 0.0001:
                    break
                iter += 1
                prevError = actError
                h.riskDesc(X, Y, lr)
                pY = h.getClassProbs(X)
                actError = np.average(1 - pY[np.arange(m), Y])
                res_it.append(actError)
                # print(f"{iter} {actError} {prevError - actError} {prevError - actError < 0.001}")
            res.append([algorithm, res_it])
        except Exception as e:
            # Handling the exception by printing its description
            print(f"Exception at iter {iter}:\n {e}")

        print(res)
        plt.figure(figsize=(10, 6))
        plt.plot(res[cont_plot][1])
        plt.title('Error vs. Iteration in ' + dataNames[d] + " lr: " + str(lr))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Error')
        plt.legend(" RD", loc="lower left")
        plt.savefig("./Results/QDA_" + dataNames[d] + ".png")
        #plt.show()
        cont_plot = cont_plot + 1


if __name__ == '__main__':
    #experiments_NB(numIter=64,lr=0.1)
    #experiments_logisticRegression(numIter=64,lr=0.1)
    #experiments_QDA(numIter= 4)

    #experiments_RF(numIter= 4)
    #experiments_logisticRegression()

    #X, Y = utl.load_cifar10_features_resnet18(with_info=False, split=False)
    #print(X.shape)
    experiment_embedded_data_NB()
    #experiments_logisticRegression()
    #experiments_QDA()
    #experiments_NB()
