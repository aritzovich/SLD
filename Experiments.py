# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt

#from matplotlib import pyplot as plt
#from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import sklearn.svm

from Models import NaiveBayesDisc, QDA, project_onto_simplex
from LogReg import LogReg
from LogRegOld import LogRegOld
from Utils import preprocess_data
import Utils as utl
import os
import warnings
# Add this line to ignore all warnings
warnings.filterwarnings("ignore")


dataNames = ['QSAR', 'adult', "climate_model", "diabetes", 'ecoli', 'german_numer', 'glass', 'haberman', 'heart',
             'indian_liver', "ionosphere", 'iris', 'letterrecog', 'liver_disorder', 'magic', 'mammographic', 'optdigits',
             'pulsar', 'redwine', 'satellite', 'sonar', 'splice', 'svmguide3', 'vehicle', 'blood_transfusion',
             "mnist", "catsvsdogs", "cifar10", "fashion_mnist", "yearbook"]
#, 'glass' falla log reg]#, 'thyroid'#QDA casca desde el principio"blood_transfusion",]
#dataNames = ['redwine','indian_liver','liver_disorder']

def experiments_logisticRegression(lr= 0.1, numIter=128, seed= 0, uniform= False):
    '''
    logistic regression using gradient descent (GD) VS using ERD

    Models:
    - GD + standard initialization (parameters=0)
    - GD + parametric initialization
    - ERD (implicit parametric initialization)

    :param dataNames: data sets
    :param numIter: numero of iterations for the iterative algorithms
    :param num_rep: number of repetitions of the experiment to account for the variability of the results
    :return:
    '''
    np.random.seed(seed)


    res = []
    classif = "LR"
    algorithms= ["RD", "GD"]
    types= ["ML"]

    for dataName in dataNames:
        print(dataName)
        print("########")
        X, Y = eval("utl.load_" + dataName + '(return_X_y=True)')
        X, Y= preprocess_data(X, Y)
        cardY = np.unique(Y).shape[0]
        m, n = X.shape

        randY= np.random.choice(cardY,m)


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

        priors = [(cardY, 1, 1)]

        for algorithm in algorithms:
            for type in types:
                for prior in priors:
                    #try:
                    iter= 1

                    if algorithm== "RD":
                        h = LogReg(n, cardY, canonical=True)
                        #hold = LogRegOld(n, cardY, canonical=True)
                    elif algorithm== "GD":
                        h = LogReg(n, cardY, canonical=False)
                        #hold = LogRegOld(n, cardY, canonical=False)

                    if type == "ML":
                        if uniform:
                            h.fit(X, randY)
                        else:
                            h.fit(X, Y)
                        #hold.fit(X,Y)
                        prior= None
                    elif type == "MAP":
                        if uniform:
                            h.fit(X, randY,  ess= prior[0], mean0= 0, w_mean0= prior[1], cov0= 1, w_cov0= prior[2])
                        else:
                            h.fit(X, Y, ess=prior[0], mean0=0, w_mean0=prior[1], cov0=1, w_cov0=prior[2])

                    pY= h.getClassProbs(X)
                    #pYold= h.getClassProbs(X)
                    prevError= np.inf
                    actError= np.average(Y != np.argmax(pY, axis=1))
                    iniError= actError
                    minError= actError
                    minIter= 1

                    # ['dataset', 'm', 'n, 'algorithm', 'iter.', 'score', 'type', 'error']
                    res.append([dataName, m, n, classif, algorithm, type, iter, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                    res.append([dataName, m, n, classif, algorithm, type, iter, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                    res.append([dataName, m, n, classif, algorithm, type, iter, 's0-1', actError])

                    print(f"classif {classif} algorithm {algorithm} based on {type} with priors {prior}")
                    print(f"iter\t actError\t descent")
                    while iter < numIter:
                        if int(np.log2(iter)) < int(np.log2(iter + 1)):
                            print(f"{iter}\t {actError}\t {prevError > actError}")
                        iter +=1

                        prevError= actError
                        if algorithm== "RD":
                            if type== "ML":
                                h.riskDesc(X,Y,lr)
                                #hold.riskDesc(X,Y,lr)
                            elif type== "MAP":
                                h.riskDesc(X, Y, lr, ess= prior[0], mean0= 0, w_mean0= prior[1], cov0= 1, w_cov0= prior[2])
                        else:
                            h.gradDesc(X,Y,lr)
                            #hold.gradDesc(X,Y,lr)
                        pY = h.getClassProbs(X)
                        #pYold= hold.getClassProbs(X)

                        actError= np.average(Y != np.argmax(pY, axis=1))

                        if actError< minError:
                            minError= actError
                            minIter= iter

                        # ['seed', 'dataset', 'm', 'n, 'algorithm', 'iter.', 'score', 'type', 'error']
                        res.append([dataName, m, n, classif, algorithm, type, iter, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                        res.append([dataName, m, n, classif, algorithm, type, iter, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                        res.append([dataName, m, n, classif, algorithm, type, iter, 's0-1', actError])
                    print(f"{iter}\t {actError}\t from {iniError} to {minError} at {minIter}")
                    #print(f"error diference: {actError - oldError}")

                    if type== "ML":
                        break

                    #except Exception as e:
                        # Handling the exception by printing its description
                    #    print(f"Exception in data {dataName} with classif {classif} algorithm {algorithm} type {type} with priors {prior} at iter {iter}:\n {e}")


    file= f"./Results/results_exper_LR_lr{lr}.csv"
    if os.path.exists(file):
        file = f"./Results/results_exper_LR_lr{lr}.csv.2"

    print(f"Saving results into "+file)
    df= pd.DataFrame(res, columns=["data", "m", "n", "classif", "alg", "type", "iter", "loss", "val"])
    df.to_csv(file, index=False)  # Set index=False if you don't want to save the index
    print(f"Results saved")

def experiments_QDA(lr= 0.1, numIter=128, seed= 0, correct_stats= True):
    '''
    Quadratic Discriminant Analysis (QDA) using gradient descent (GD) VS using risk-based calibration (RC)

    Models:
    - GD + standard initialization (parameters=0)
    - GD + parametric initialization
    - ERD (implicit parametric initialization)

    :param lr: learning rate for gradient descent
    :param numIter: numero of iterations for the iterative algorithms
    :return:
    '''

    np.random.seed(seed)

    #dataNames = ['forestcov']
    res = []
    classif = "QDA"
    algorithms= ["RD","GD"]
    types= ["ML", "MAP"]

    for dataName in dataNames:
        X, Y = eval("utl.load_" + dataName + '(return_X_y=True)')
        X, Y= preprocess_data(X, Y)
        cardY = np.unique(Y).shape[0]
        m, n = X.shape
        priors = [(cardY, 10, 10)]

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


        h= QDA(cardY, n)

        for algorithm in algorithms:
            for type in types:
                for prior in priors:
                    try:
                        if type == 'ML':
                            h.fit(X,Y)
                            prior= None
                        elif type == 'MAP':
                            h.fit(X,Y,ess= prior[0], mean0= 0, w_mean0=prior[1], cov0= 1, w_cov0= prior[2])

                        pY= h.getClassProbs(X)
                        prevError= np.inf
                        actError=  np.average(Y != np.argmax(pY, axis=1))
                        iniError= actError
                        minError= actError
                        minIter= 1

                        # ['dataset', 'm', 'n, 'algorithm', 'type', 'iter', 'score', 'type', 'error']
                        res.append([dataName, m, n, classif, algorithm, type, 1, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                        res.append([dataName, m, n, classif, algorithm, type, 1, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                        res.append([dataName, m, n, classif, algorithm, type, 1, 's0-1', np.average(1- pY[np.arange(m), Y])])

                        iter= 1
                        print(f"classif {classif} algorithm {algorithm} based on {type} with priors {prior}")
                        print(f"iter\t actError\t descent")
                        while iter < numIter:
                            if int(np.log2(iter)) < int(np.log2(iter + 1)):
                                print(f"{iter}\t {actError}\t {prevError > actError}")
                            iter +=1
                            prevError= actError
                            if algorithm== "RD":
                                if type== 'ML':
                                    h.riskDesc(X,Y,lr,correct_forbidden_stats= correct_stats)
                                else:
                                    #h.riskDesc(X,Y,lr,ess= cardY, mean0= 0, w_mean0=cardY, cov0= 1, w_cov0= cardY)
                                    h.riskDesc(X,Y,lr,ess= prior[0], mean0= 0, w_mean0=prior[1], cov0= 1, w_cov0= prior[2],
                                               correct_forbidden_stats= correct_stats)
                            elif algorithm== "GD":
                                h.gradDesc(X,Y,lr)

                            pY = h.getClassProbs(X)
                            if not np.isclose(np.sum(np.sum(pY, axis= 1))/m, 1, atol=10**-6):
                                print(f"The class probabilities does not sum one: {np.sum(np.sum(pY, axis= 1))/m}")
                                h.getClassProbs(X)

                            actError=  np.average(Y != np.argmax(pY, axis=1))
                            if actError< minError:
                                minError= actError
                                minIter= iter
                            # ['seed', 'dataset', 'm', 'n, 'algorithm', 'iter.', 'score', 'type', 'error']
                            res.append([dataName, m, n, classif, algorithm, type, iter, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                            res.append([dataName, m, n, classif, algorithm, type, iter, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                            res.append([dataName, m, n, classif, algorithm, type, iter, 's0-1', np.average(1- pY[np.arange(m), Y])])
                        print(f"{iter}\t {actError}\t from {iniError} to {minError} at {minIter}")

                    except Exception as e:
                        # Handling the exception by printing its description
                        print(f"Exception {e} in data {dataName}")
                        for rest_iter in range(iter,numIter+1):
                            res.append([dataName, m, n, classif, algorithm, type, rest_iter, '0-1',
                                        np.average(Y != np.argmax(pY, axis=1))])
                            res.append([dataName, m, n, classif, algorithm, type, rest_iter, 'log',
                                        np.average(- np.log(pY[np.arange(m), Y]))])
                            res.append([dataName, m, n, classif, algorithm, type, rest_iter, 's0-1',
                                        np.average(1 - pY[np.arange(m), Y])])
                        print(f"{iter}\t {actError}\t from {iniError} to {minError} at {minIter}")

                    if type == 'ML':
                      # There is no prior
                      break

    file= f"./Results/results_exper_QDA_lr{lr}.csv"
    if os.path.exists(file):
        file = f"./Results/results_exper_QDA_lr{lr}.csv.2"

    print(f"Saving results into "+file)
    df= pd.DataFrame(res, columns=["data", "m", "n", 'classif', "alg", 'type', "iter", "loss", "val"])
    df.to_csv(file, index=False)  # Set index=False if you don't want to save the index
    print(f"Results saved")

def experiments_NB(lr= 0.1, numIter=128, seed= 0):
    '''
    logistic regression using gradient descent (GD) VS using ERD, using maximum likelihood (ML) and maximum a
    posteriori with uniform Dirichlet prior with alpha= card(Y) (MAP)

    Models:
    - GD + standard initialization (parameters=0)
    - GD + parametric initialization
    - ERD (implicit parametric initialization)

    :param dataNames: data sets
    :param numIter: numero of iterations for the iterative algorithms
    :param num_rep: number of repetitions of the experiment to account for the variability of the results
    :return:
    '''

    np.random.seed(seed)

    res = []
    classif = "NB"
    algorithms= ["RD", "GD", "DFE"]
    types= ["ML", "MAP"]
    card= 5
    corrections= [0,2]

    for dataName in dataNames:
        X, Y = eval("utl.load_" + dataName + '(return_X_y=True)')
        # Remove infrequent classes, constant columns and normalize
        #card= int(np.log(X.shape[0]))
        card = 5

        X, Y = preprocess_data(X, Y, n_bins= card)
        m, n = X.shape
        cardY = np.unique(Y).shape[0]

        ess = cardY

        print(dataName)
        print("########")
        print("(m,n)=" + str((m, n)))
        print("card Y= " + str(cardY))
        print("proportions: " + str(np.unique(Y, return_counts=True)[1]))
        # Set printing options
        np.set_printoptions(precision=2, suppress=True)
        # Display the number of unique values in each row
        print("Number of unique values in each row:")
        print(np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X))

        for alg in algorithms:
            for type in types:
                try:
                    h= NaiveBayesDisc(cardY, card, n)

                    if type== "ML":
                        h.fit(X, Y)
                    elif type== "MAP":
                        h.fit(X, Y, ess= ess)

                    pY = h.getClassProbs(X)
                    prevError = np.inf
                    actError=  np.average(Y != np.argmax(pY, axis=1))
                    initError= actError
                    minError= actError
                    minIter= 1

                    # ['seed', 'dataset', 'm', 'n, 'classif', 'algorithm', 'iter.', 'score', 'type', 'error']
                    res.append([dataName, m, n, classif, alg, type, 1, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                    res.append([dataName, m, n, classif, alg, type, 1, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                    res.append([dataName, m, n, classif, alg, type, 1, 's0-1', np.average(1- pY[np.arange(m), Y])])


                    iter = 1
                    #print(f"classif {classif} algorithm {alg} based on {type}")
                    print(f"classifier {classif} algorithm {alg} based on {type}")
                    print(f"iter actError     descend")
                    while iter < numIter:
                        if int(np.log2(iter)) < int(np.log2(iter + 1)):
                            print(f"{iter}\t{actError}\t{prevError > actError}")
                        iter += 1
                        prevError = actError
                        if alg== "RD":
                            if type == "ML":
                                h.riskDesc(X, Y, lr)
                            elif type == "MAP":
                                h.riskDesc(X, Y, lr, ess= ess)
                        elif alg== "GD":
                            h.gradDesc(X, Y, lr)
                        elif alg== "DFE":
                            if type == "ML":
                                h.DFE(X, Y, lr)
                            elif type == "MAP":
                                h.DFE(X, Y, lr, ess= ess)

                        pY = h.getClassProbs(X)

                        #print(f"nans in \tpy {np.isnan(pY_).any()}\t in log py {np.isnan(pY).any()}")

                        actError=  np.average(Y != np.argmax(pY, axis=1))
                        if minError>actError:
                            minError= actError
                            minIter= iter
                        # ['seed', 'dataset', 'm', 'n, 'algorithm', 'iter.', 'score', 'type', 'error']
                        res.append([dataName, m, n, classif, alg, type, iter, '0-1', np.average(Y != np.argmax(pY, axis=1))])
                        res.append([dataName, m, n, classif, alg, type, iter, 'log', np.average(- np.log(pY[np.arange(m), Y]))])
                        res.append([dataName, m, n, classif, alg, type, iter, 's0-1', np.average(1- pY[np.arange(m), Y])])

                    print(f"{iter}\t {actError} from {initError} min {minError} at iter {minIter}")
                except Exception as e:
                    # Handling the exception by printing its description
                    print(f"Exception in data {dataName} with algorithm {alg} type {type} at iter {iter}:\n {e}")

    file= f"./Results/results_exper_NB_lr{lr}.csv"
    if os.path.exists(file):
        file = f"./Results/results_exper_NB_lr{lr}.csv.2"
    print(f"Saving results into "+file)
    df= pd.DataFrame(res, columns=["data", "m", "n", "classif", "alg", "type", "iter", "loss", "val"])
    df.to_csv(file, index=False)
    print(f"Results saved")


if __name__ == '__main__':
    experiments_QDA(numIter= 64)
#    experiments_NB(numIter=64)
#    experiments_logisticRegression(numIter=64)

