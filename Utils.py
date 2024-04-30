import zipfile
from os.path import join, dirname
import csv
import numpy as np
from sklearn.datasets._base import Bunch, load_files, load_csv_data#load_data,
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

import pandas as pd


def scaling_pY(pY):
    '''
    linear scaling and normalization of the array of clas probability distributions: avoid probs >1 and <0

    :param pY: Array of missespecified class distributions: They could contain negative values and/or distributions
    with a probability mass larger than 1. Number of instances x number of class labels
    :return: an array with proper class distributions
    '''

    size, cardY= pY.shape

    down = np.min(np.column_stack([np.min(pY, axis=1), np.zeros(size)]), axis=1)
    up = np.max(np.column_stack([np.max(pY, axis=1), np.ones(size)]), axis=1)
    pY = (pY - np.repeat(down, cardY).reshape((size, cardY))) / np.repeat(up - down, cardY).reshape((size, cardY))

    return pY

def preprocess_data(X,Y,min_cond_var= 10**-5, discretize= False):

    m,n= X.shape

    # Remove outlier classes (a frequency less than 5%)
    v, f = np.unique(Y, return_counts=True)
    remove = v[(f< 10) & (f < m * 0.05)]
    for c in remove:
        X = X[Y != c]
        Y = Y[Y != c]


    # Find rows with NaNs or infinity values in matrix X
    rows_to_remove = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)

    # Remove corresponding rows in matrix X and elements in vector Y
    X = X[~rows_to_remove]
    Y = Y[~rows_to_remove]

    # Map the values of Y from 0 to r-1
    domY = np.unique(Y)
    Y_proc = np.zeros(Y.shape[0], dtype=int)
    for i, y in enumerate(domY):
        Y_proc[Y == y] = i

    if not discretize:
        cardY= len(np.unique(Y))
        # remove variables with low variance given any class label
        del_vars= np.zeros(n,dtype=bool)
        for c in range(cardY):
            del_vars[np.var(X[Y_proc==c], axis= 0)< min_cond_var]= True

        X = X[:, del_vars == False]

        # Calculate mean and standard deviation
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # Normalize columns to have zero mean and variance 1
        X_normalized = (X - mean) / std

        return X_normalized, Y_proc
    else:
        if discretize:
            # Initialize KBinsDiscretizer with strategy='quantile' for equal frequency binning
            # n_bins specifies the number of bins
            discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')

            # Fit and transform the data
            X_disc = discretizer.fit_transform(X).astype(int)
            unique_counts = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0, arr=X_disc)
            if np.any(unique_counts < 2):
                X_disc = X_disc[:, unique_counts == 2]
        return X_disc, Y_proc

def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y


def getDataNames():
    return ['adult',
               'iris',
               'optdigits',#riskDesc gana y aguanta al menos 10000 iteraciones
               'satellite',
               'vehicle',
               'segment',
               'redwine',
               'letterrecog',
               'forestcov',
               'ecoli',
               'credit',
               'magic',
               'diabetes',
               'glass',
               'haberman',
               'mammographic',
               'indian_liver',
               'heart',
               'sonar',
               'svmguide3',
               'liver_disorder',
               'german_numer',
               "blood_transfusion",
               "climate_model",
               "diabetes",
               "ionosphere",
               'pulsar',
               'QSAR',
                'splice']



def load_mnist_features_resnet18(with_info=False, split=False):
    """Load and return the MNIST Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ======================= ===========================
    Classes                                          2
    Samples per class Train [5923,6742,5958,6131,5842,
                             5421,5918,6265,5851,5949]
    Samples per class Test    [980,1135,1032,1010,982,
                                892,958,1028,974,1009]
    Samples total Train                          60000
    Samples total Test                           10000
    Samples total                                70000
    Dimensionality                                 512
    Features                                     float
    ======================= ===========================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    split : boolean, default=False.
        If True, returns a dictionary instead of an array in the place of the
        data.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of MNIST ResNet18 features
        csv dataset. If `split=False`, data is
        an array. If `split=True` data is a dictionary with 'train' and 'test'
        splits.

    (data, target) : tuple if ``with_info`` is False. If `split=False`, data is
        an array. If `split=True` data is a dictionary with 'train' and 'test'
        splits.
    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'mnist_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('mnist_features_resnet18_1.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('mnist_features_resnet18_2.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_3.csv.zip'))
    df3 = pd.read_csv(zf.open('mnist_features_resnet18_3.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_4.csv.zip'))
    df4 = pd.read_csv(zf.open('mnist_features_resnet18_4.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_5.csv.zip'))
    df5 = pd.read_csv(zf.open('mnist_features_resnet18_5.csv'), header=None)

    dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    if not with_info:
        if split:
            # X_train, X_test, Y_train, Y_test
            X_train = data[:60000, :]
            Y_train = target[:60000]
            X_test = data[60000:, :]
            Y_test = target[60000:]
            return X_train, X_test, Y_train, Y_test
        else:
            return data, target
    else:
        if split:
            data = {'train': data[:60000, :], 'test': data[60000:, :]}
            target = {'train': target[:60000], 'test': target[60000:]}
        return Bunch(data=data, target=target, DESCR=descr_text)
    return 0


def load_catsvsdogs_features_resnet18(with_info=False):
    """Load and return the Cats vs Dogs Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ===========================================
    Classes                                   2
    Samples per class             [11658,11604]
    Samples total                         23262
    Dimensionality                          512
    Features                              float
    ===========================================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of Cats vs Dogs ResNet18 features
        csv dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr',
                       'catsvsdogs_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'catsvsdogs_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_1.csv'))
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'catsvsdogs_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_2.csv'))

    dataset = np.array(pd.concat([df1, df2]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=descr_text)


def load_yearbook_features_resnet18(with_info=False, with_attributes=False):
    """Load and return the Yearbook Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ===========================================
    Classes                                   2
    Samples per class             [20248,17673]
    Samples total                         37921
    Dimensionality                          512
    Features                              float
    ===========================================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    with_attributes : boolean, default=False.
        If True, returns an additional dictionary containing information of
        additional attributes: year, state, city, school of the portraits.
        The key 'attr_labels' in the dictionary contains these labels
        corresponding to each columns, while 'attr_data' corresponds to
        the attribute data in form of numpy array.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of Yearbook ResNet18 features
        csv dataset.

    (data, target) : tuple if ``with_info`` is False

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'yearbook_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'yearbook_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('yearbook_features_resnet18_1.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'yearbook_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('yearbook_features_resnet18_2.csv'), header=None)

    dataset = np.array(pd.concat([df1, df2]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if with_attributes:
        attr = pd.read_csv(join(module_path, 'data',
                                'yearbook_attributes.csv'))
        attr_labels = attr.columns.values
        attr_val = attr.values
        attr = {'attr_labels': attr_labels, 'attr_data': attr_val}

        if not with_info:
            return data, normalizeLabels(target), attr

        return Bunch(data=data, target=normalizeLabels(target),
                     attributes=attr, DESCR=descr_text)

    else:
        if not with_info:
            return data, normalizeLabels(target)

        return Bunch(data=data, target=normalizeLabels(target),
                     DESCR=descr_text)





def load_thyroid(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)


    data_file_name = join(module_path, 'data/multi_class_datasets/', 'thyroid.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR="The data set has no descripsion file .rst",
                 filename=data_file_name)

def load_splice(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'splice.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'splice.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)

def load_QSAR(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'QSAR.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'QSAR.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)


def load_pulsar(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    #fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'pulsar.rst')
    #with open(fdescr_name) as f:
    #    descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'pulsar.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR="The data set has no descripsion file .rst",
                 filename=data_file_name)

def load_ionosphere(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'ionosphere.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'ionosphere.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)

def load_diabetes(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)

def load_climate_model(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'climate_model.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'climate_model.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)

def load_blood_transfusion(return_X_y=False):
    """
    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'blood_transfusion.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'blood_transfusion.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = list()
        target = list()

        for i, d in enumerate(data_file):
            data.append(np.asarray(d[:-1], dtype=float))
            target.append(int(d[-1]))

        data = np.array(data)
        target = np.array(target)
        feature_names = np.array([i for i in range(data.shape[1])])

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names,
                 DESCR=descr_text,
                 filename=data_file_name)

def load_adult(return_X_y=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    
    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'adult.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype= int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_diabetes(return_X_y=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_iris(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 3
    Samples per class              [50,50,50]
    Samples total                         150
    Dimensionality                          4
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'iris.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'iris.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype= int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype= int)
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_redwine(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class            [1599, 4898]
    Samples total                        6497
    Dimensionality                         11
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'redwine.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'redwine.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray([float(i) for i in d[:-1]], dtype=float)
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_forestcov(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 7
    Samples per class [211840,283301,35754,
                     2747,9493,17367,20510,0]
    Samples total                      581012
    Dimensionality                         54
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'forestcov.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'forestcov.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype= int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_letterrecog(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                26
    Samples total                       20000
    Dimensionality                         16
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'letter-recognition.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'letter-recognition.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[1:], dtype=float)
            if d[0] in classes:
                index = classes.index(d[0])
                target[i] = np.asarray(index, dtype= int)
            else:
                classes.append(d[0])
                target[i] = np.asarray(classes.index(d[0]), dtype= int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_ecoli(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 8
    Samples per class [143,77,52,35,20,5,2,2]
    Samples total                         336
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'ecoli.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'ecoli.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray([float(i) for i in d[1:-1]], dtype=float)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype= int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_vehicle(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 4
    Samples per class       [240,240,240,226]
    Samples total                         846
    Dimensionality                         18
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'vehicle.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'vehicle.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype= int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype= int)

    if return_X_y:
        return data, target

def load_segment(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 7
    Samples per class              [383, 307]
    Samples total                        2310
    Dimensionality                         19
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'segment.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'segment.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray([float(i) for i in d[:-1]], dtype=float)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_satellite(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 6
    Samples per class               383, 307]
    Samples total                        6435
    Dimensionality                         36
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'satellite.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'satellite.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=float)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_optdigits(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class               383, 307]
    Samples total                        5620
    Dimensionality                         64
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi_class_datasets/', 'optdigits.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'optdigits.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=float)# remove the first variable (is a constant)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data[:,np.array([len(np.unique(data[:,i])) for i in range(n_features)])>1], target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_credit(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               383, 307]
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'credit.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype= int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=float)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype= int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_magic(return_X_y=False):
    """Load and return the Magic Gamma Telescope dataset (classification).

    =========================================
    Classes                                 2
    Samples per class            [6688,12332]
    Samples total                       19020
    Dimensionality                         10
    Features                            float
    =========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary_class_datasets/', 'magic.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'magic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype=str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_glass(return_X_y=False):
    """Load and return the Glass Identification Data Set (classification).

    ===========================================
    Classes                                   6
    Samples per class    [70, 76, 17, 29, 13, 9]
    Samples total                           214
    Dimensionality                            9
    Features                              float
    ===========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of glass csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    #data, target, target_names = load_csv_data(module_path, 'glass.csv')
    with open(join(module_path, 'descr/multi_class_datasets/', 'glass.rst')) as rst_file:
        descr_text = rst_file.read()

    data_file_name = join(module_path, 'data/multi_class_datasets/', 'glass.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype=str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['RI: refractive index',
                                "Na: Sodium (unit measurement: "
                                "weight percent in corresponding oxide, "
                                "as are attributes 4-10)",
                                'Mg: Magnesium ',
                                'Al: Aluminim',
                                'Si: Silicon',
                                'K: Potassium',
                                'Ca: Calcium',
                                'Ba: Barium',
                                'Fe: Iron'],
                 DESCR= descr_text,
                 filename=data_file_name)

def load_haberman(return_X_y=False):
    """Load and return the Haberman's Survival Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [225, 82]
    Samples total              306
    Dimensionality               3
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of haberman csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    with open(join(module_path, 'descr/binary_class_datasets/', 'haberman.rst')) as rst_file:
        descr_text = rst_file.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'haberman.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype=str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['PatientAge', 
                                'OperationYear', 
                                'PositiveAxillaryNodesDetected'],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_mammographic(return_X_y=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    with open(join(module_path, 'descr/binary_class_datasets/', 'mammographic.rst')) as rst_file:
        descr_text = rst_file.read()

    data_file_name = join(module_path, 'data/binary_class_datasets/', 'mammographic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=float)
            target[i] = np.asarray(d[-1], dtype=str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_indian_liver(return_X_y=False):
    """Load and return the Indian Liver Patient Data Set 
    (classification).

    =========================================================
    Classes                                                 2
    Samples per class                              [416, 167]
    Samples total                                         583
    Dimensionality                                         10
    Features                                       int, float
    Missing Values                                     4 (nan)
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    """data, target, target_names = load_data(module_path,
                                           'indianLiverPatient.csv')"""
    with open(join(module_path, 'data/binary_class_datasets/',
                   'indianLiverPatient.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype= int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=float)
            target[i] = np.asarray(ir[-1], dtype= int)
    with open(join(module_path, 'descr/binary_class_datasets/',
                   'indianLiverPatient.rst')) as rst_file:
        descr_text = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=descr_text,
                 feature_names=['Age of the patient',
                                'Gender of the patient',
                                'Total Bilirubin',
                                'Direct Bilirubin',
                                'Alkaline Phosphotase',
                                'Alamine Aminotransferase',
                                'Aspartate Aminotransferase',
                                'Total Protiens',
                                'Albumin',
                                'A/G Ratio'])

def load_heart(return_X_y=False):
    """Load and return the Heart Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         270
    Dimensionality                                         13
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data/binary_class_datasets/',
                   'heart.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(270)
        n_features = int(13)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=float)
            target[i] = np.asarray(ir[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_sonar(return_X_y=False):
    """Load and return the Sonar Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         208
    Dimensionality                                         60
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data/binary_class_datasets/',
                   'sonar.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(208)
        n_features = int(60)
        data = []
        target = []

        for i, ir in enumerate(data_file):
            # print(float(ir[-1]))
            if len(ir[1:]) == 60:
                data.append(np.asarray(ir[1:], dtype=float))
                target.append(int(ir[0]))
        data = np.asarray(data)
        target = np.asarray(target)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_svmguide3(return_X_y=False):
    """Load and return the SVM guide Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                        1243
    Dimensionality                                         21
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data/binary_class_datasets',
                   'svmguide3.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(1243)
        n_features = int(21)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:22], dtype=float)
            target[i] = np.asarray(ir[0], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_liver_disorder(return_X_y=False):
    """Load and return the Liver Disorder Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         345
    Dimensionality                                          5
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data/binary_class_datasets',
                   'liver_disorder.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(345)
        n_features = int(5)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=float)
            target[i] = np.asarray(int(float(ir[-1])), dtype=float)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_german_numer(return_X_y=False):
    """Load and return the German numer Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                        1000
    Dimensionality                                         24
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data/binary_class_datasets',
                   'german_numer.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(1000)
        n_features = int(24)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:], dtype=float)
            target[i] = np.asarray(ir[0], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])