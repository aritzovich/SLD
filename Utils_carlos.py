
def load_mnist(return_X_y=False, split=False):
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

    #fdescr_name = join(module_path, 'descr', 'mnist_features_resnet18.rst')
    #with open(fdescr_name) as f:
    #    descr_text = f.read()

    #zf = zipfile.ZipFile(join(module_path, 'data',
    #                          'mnist_features_resnet18_1.csv.zip'))
    # df1 = pd.read_csv(zf.open('mnist_features_resnet18_1.csv'), header=None)
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'mnist_features_resnet18_2.csv.zip'))
    # df2 = pd.read_csv(zf.open('mnist_features_resnet18_2.csv'), header=None)
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'mnist_features_resnet18_3.csv.zip'))
    # df3 = pd.read_csv(zf.open('mnist_features_resnet18_3.csv'), header=None)
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'mnist_features_resnet18_4.csv.zip'))
    # df4 = pd.read_csv(zf.open('mnist_features_resnet18_4.csv'), header=None)
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'mnist_features_resnet18_5.csv.zip'))
    # df5 = pd.read_csv(zf.open('mnist_features_resnet18_5.csv'), header=None)

    df1 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'mnist_features_resnet18_train_1.csv'), header=None, index_col=0)
    df2 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'mnist_features_resnet18_train_2.csv'), header=None, index_col=0)
    df3 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'mnist_features_resnet18_train_3.csv'), header=None, index_col=0)
    df4 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'mnist_features_resnet18_train_4.csv'), header=None, index_col=0)
    df5 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'mnist_features_resnet18_test.csv'), header=None, index_col=0)

    dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    if return_X_y:
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


def load_catsvsdogs(return_X_y=False, split=False):
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

    # fdescr_name = join(module_path, 'descr',
    #                    'catsvsdogs_features_resnet18.rst')
    # with open(fdescr_name) as f:
    #     descr_text = f.read()
    #
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'catsvsdogs_features_resnet18_1.csv.zip'))
    # df1 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_1.csv'))
    # zf = zipfile.ZipFile(join(module_path, 'data',
    #                           'catsvsdogs_features_resnet18_2.csv.zip'))
    # df2 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_2.csv'))

    df1 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'catsvsdogs_features_resnet18_1.csv'), header=None, index_col=0)
    df2 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'catsvsdogs_features_resnet18_2.csv'), header=None, index_col=0)

    dataset = np.array(pd.concat([df1, df2]))
    data = dataset[:, :-1]
    target = dataset[:, -1]
    print(data[1,1])
    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)
    target = normalizeLabels(target)

    if return_X_y:
        if split:
            # X_train, X_test, Y_train, Y_test
            X_train = data[:20000, :]
            Y_train = target[:20000]
            X_test = data[20000:, :]
            Y_test = target[20000:]
            return X_train, X_test, Y_train, Y_test
        else:
            return data, target
    else:
        if split:
            data = {'train': data[:20000, :], 'test': data[20000:, :]}
            target = {'train': target[:20000], 'test': target[20000:]}
        return Bunch(data=data, target=target, DESCR=descr_text)


def load_cifar10(return_X_y=False, split=False):
    """Load and return the CIFAR10 Data Set features extracted using a
    pretrained ResNet18 neural network (classification).
    """
    module_path = dirname(__file__)

    df1 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'cifar10_features_resnet18_train_1.csv'), header=None, index_col=0)
    df2 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'cifar10_features_resnet18_train_2.csv'), header=None, index_col=0)
    df3 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'cifar10_features_resnet18_train_3.csv'), header=None, index_col=0)
    df4 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'cifar10_features_resnet18_test.csv'), header=None, index_col=0)

    dataset = np.array(pd.concat([df1, df2, df3, df4]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    return data, target


def load_fashion_mnist(return_X_y=False, split=False):
    """Load and return the CIFAR10 Data Set features extracted using a
    pretrained ResNet18 neural network (classification).
    """
    module_path = dirname(__file__)

    df1 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'fashion_mnist_features_resnet18_train_1.csv'), header=None, index_col=0)
    df2 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'fashion_mnist_features_resnet18_train_2.csv'), header=None, index_col=0)
    df3 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'fashion_mnist_features_resnet18_train_3.csv'), header=None, index_col=0)
    df4 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'fashion_mnist_features_resnet18_train_3.csv'), header=None, index_col=0)
    df5 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'fashion_mnist_features_resnet18_test.csv'), header=None, index_col=0)

    dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    return data, target


def load_yearbook(return_X_y=False, split=False):
    """Load and return the Yearbook Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ===========================================
    Classes                                   2
    Samples per class             [20248,17673]
    Samples total                         37921
    Dimensionality                          512
    Features                              float
    ===========================================
    Returns
    -------
    (data, target) : tuple if ``with_info`` is False
    """

    module_path = dirname(__file__)

    df1 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'yearbook_features_resnet18_1.csv'),
                      header=None, index_col=0)
    df2 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'yearbook_features_resnet18_2.csv'),
                      header=None, index_col=0)
    df3 = pd.read_csv(join(module_path, 'data/embedding_datasets', 'yearbook_features_resnet18_3.csv'),
                      header=None, index_col=0)

    dataset = np.array(pd.concat([df1, df2, df3]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    return data, target