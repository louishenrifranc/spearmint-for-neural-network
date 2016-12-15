import cPickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import theano
import lasagne.nonlinearities
import json


def load_dataset():
    # f = open('dataLouis.pickle', 'rb')
    # X = cPickle.load(f)
    # y = cPickle.load(f)
    # f.close()
    X, y = cPickle.load(open('dataLouis.pickle', 'rb'))
    nb_example = len(X)

    s1 = int(0.6 * nb_example)
    s2 = int(0.8 * nb_example)
    X_train, y_train = X[:s1, ], y[:s1]
    X_val, y_val = X[s1:s2, ], y[s1:s2]
    X_test, y_test = X[s2:, ], y[s2:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def pca_reduction(nb_components):
    X, y = cPickle.load(open('dataLouis.pickle', 'rb'))
    ss = StandardScaler()
    X_train_std = ss.fit_transform(X)
    pca = PCA(
        n_components=nb_components)  # http://stats.stackexchange.com/questions/123318/why-are-there-only-n-1-principal-components-for-n-data-points-if-the-number
    X_train_pca = pca.fit_transform(X_train_std)
    print(len(X_train_pca[0]))
    f = open("dataLouis_smaller.pickle", "w")
    cPickle.dump(X_train_pca, f)
    cPickle.dump(y, f)
    f.close()
    return pca.explained_variance_ratio_


def get_dtype():
    return 'float16'


def get_weights(name, n_in, n_out=None):
    """
    Create a 2D np array using specific initialization
    Parameters
    ----------
    name: string
        Initialization name
    n_in: int
        Number of rows
    n_out: int (default None for biais only)
        Number of columns
    :return: a np.array
    """
    if name == "randU":
        return np.random.uniform(-0.01, 0.01, size=(n_in, n_out))
    elif name == "randN":
        return np.random.normal(1, 0.1, size=(n_in, n_out))
    elif name == "glorotU":
        return np.random.uniform(low=-4. / np.sqrt(6.0 / (n_in + n_out)),
                                 high=4. / np.sqrt(6.0 / (n_in + n_out)), size=(n_in, n_out))
    elif name == "zeros":
        if n_out is None:
            # For biais
            return np.zeros((n_in))
        else:
            return np.zeros((n_in, n_out))
    else:
        raise 'Unsupported weigh init %s' % name


def get_optimizer(name, loss, params, lr, decay_lr):
    '''
    Get a lasagne optimizer object
    Parameters
    ----------
    name: string
        Name of the optimizer
    loss:
        Loss function to minimize
    params:
        Params to updates
    :param lr: scalar or theano.tensor.scalar
        Learning rate
    :return: a lasagne.updates
    '''
    if name == "adadelta":
        return lasagne.updates.adadelta(loss, params)
    elif name == "adagrad":
        return lasagne.updates.adagrad(loss, params, learning_rate=lr)
    elif name == "rmsprop":
        return lasagne.updates.rmsprop(loss, params, learning_rate=lr, rho=decay_lr)


def get_nonlinearity(name):
    '''
    Get a lasagne nonlinearities object
    Parameters
    ----------
    name: string (default : tanh)
        Nonlinearity function
    :return: a lasagne.nonlinearities
    '''
    if name == "relu":
        return lasagne.nonlinearities.rectify
    elif name == "tanh":
        return lasagne.nonlinearities.tanh
    elif name == "sigmoid":
        return lasagne.nonlinearities.sigmoid
    elif name == "linear":
        return lasagne.nonlinearities.linear
    else:
        raise 'Unsupported activation function %s' % name


def iterate_minibatches(inputs, outputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], outputs[excerpt]


def get_shared(name, n_in, n_out, borrow=True):
    """
    Create a theano.shared 2D array
    Parameters
    ----------
    name: string
        Initialization of the element in the tensor
    n_in: int
        Number of rows
    n_out: int
        Number of columns
    borrow: boolean
        Shared the tensor
    :return: a theano.shared
    """
    return theano.shared(get_weights(name, n_in, n_out).astype(get_dtype()), borrow=borrow)


def get_nn_parameters(filename='data/global_nn_parameters.json'):
    """
    Return neural network attributes in a dic
    Parameters
    ----------
    filename: string (default: data/global_nn_parameters.json)
        file containing the attributes of the neural network

    :return: a dict containing the optimizer,
                             the batch size,
                             the number of input,
                             the learning rate
    """
    return json.load(open(filename, 'rb'))
