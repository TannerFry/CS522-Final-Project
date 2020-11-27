import numpy as np

def euc2(x, y):
    # calculate squared Euclidean distance

    # check dimension
    assert x.shape == y.shape

    diff = x - y

    return np.dot(diff, diff)


def mah2(x, y, Sigma):
    # calculate squared Mahalanobis distance

    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    
    diff = x - y
    
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)


def gaussian(x, y, Sigma):
    # multivariate Gaussian

    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)

    d = max(x.shape)          # dimension
    dmah2 = mah2(x, y, Sigma)
    gx = 1.0 / ((2*np.pi)**(d/2) * np.linalg.det(Sigma)**0.5) * np.exp(-0.5 * dmah2)
    return gx


def accuracy_score(y, y_model):
    # calculate classification overall accuracy and classwise accuracy
    
    assert len(y) == len(y_model)
    classn = len(np.unique(y))       # number of different classes
    correct_all = y == y_model       # all correct classifications
    acc_overall = np.sum(correct_all) / len(y)
    acc_i = np.zeros(classn)
    for i in range(classn):   
        GT_i = y == i                # samples actually belong to class i
        acc_i[i] = (np.sum(GT_i & correct_all) / np.sum(GT_i))
        
    return acc_i, acc_overall


def load_data(f):
    """ Assume data format:
    feature1 feature 2 ... label 
    """

    # process training data
    data = np.genfromtxt(f)
    
    # return all feature columns except last
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    return X, y


def normalize(Tr, Te = None):
    # normalize the dataset such that different dimensions would have the same scale
    # use statistics of the training data to normalize both the training and testing sets
    # if only one argument, then just normalize that one set
    
    ntr, _ = Tr.shape
    stds = np.std(Tr, axis = 0)

    normTr = (Tr - np.tile(np.mean(Tr, axis = 0), (ntr, 1))) / stds
    if Te is not None:
        nte, _ = Te.shape
        normTe = (Te - np.tile(np.mean(Tr, axis = 0), (nte, 1))) / stds
    
    if Te is not None:
        return normTr, normTe
    else:
        return normTr
