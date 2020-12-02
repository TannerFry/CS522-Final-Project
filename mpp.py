#
# Maximum Posterior Probability (MPP)
#
#    Supervised parametric learning assuming Gaussian pdf
#    with 3 cases of discriminant functions
#
#    Sample code for the Machine Learning class at UTK
#
# Hairong Qi, hqi@utk.edu
#
import numpy as np
import sys
import time
import util
import load_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def mpp(Tr, yTr, Te, cases, P):
    # training process - derive the model
    covs, means = {}, {}     # dictionaries
    covsum = None

    classes = np.unique(yTr)   # get unique labels as dictionary items
    print(f"classes = {classes}")
    classn = len(classes)    # number of classes
    print(f"classn = {classn}")
    
    for c in range(classn):
        # filter out samples for the c^th class
        arr = Tr[yTr == classes[c]]  
        arr = arr.astype(float)
        print(f"arr.shape = {arr.shape}")
        # calculate statistics
        covs[c] = np.cov(np.transpose(arr))
        means[c] = np.mean(arr, axis=0)  # mean along the columns
        # accumulate the covariance matrices for Case 1 and Case 2
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]
    
    # used by case 2
    covavg = covsum / classn
    # used by case 1
    varavg = np.sum(np.diagonal(covavg)) / classn
            
    # testing process - apply the learned model on test set 
    disc = np.zeros(classn)
    nr, _ = Te.shape
    y = np.zeros(nr)            # to hold labels assigned from the learned model

    for i in range(nr):
        for c in range(classn):
            if cases == 1:
                edist2 = util.euc2(means[c], Te[i])
                disc[c] = -edist2 / (2 * varavg) + np.log(P[c] + 0.000001)
            elif cases == 2: 
                mdist2 = util.mah2(means[c], Te[i], covavg)
                disc[c] = -mdist2 / 2 + np.log(P[c] + 0.000001)
            elif cases == 3:
                mdist2 = util.mah2(means[c], Te[i], covs[c])
                disc[c] = -mdist2 / 2 - np.log(np.linalg.det(covs[c])) / 2 + np.log(P[c] + 0.000001)
            else:
                print("Can only handle case numbers 1, 2, 3.")
                sys.exit(1)
        y[i] = disc.argmax()
            
    return y

def main():

    # load data
    training_data = load_data.read_data("train.csv")
    testing_data = load_data.read_data("test.csv")
    testing_labels = load_data.read_data("submission.csv")
    X_train, X_test = load_data.vectorize_data(training_data, testing_data)

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    
    Y_train = np.array(training_data)[:, -1]
    Y_test = np.array(testing_labels)[:, -1]
    
    # the training and testing datasets should have the same dimension
    _, nftrain = X_train.shape
    _, nftest = X_test.shape
    assert nftrain == nftest   
    
    # ask the user to input which discriminant function to use
    prompt = '''
    Type of discriminant functions supported assuming Gaussian pdf:
    1 - minimum Euclidean distance classifier
    2 - minimum Mahalanobis distance classifier
    3 - quadratic classifier
    '''
    print(prompt)
    str = input('Please input 1, 2, or 3: ')
    cases = int(str)

    # ask the user to input prior probability that needs to sum to 1
    prop_str = input("Please input prior probabilities in float numbers, separated by space, and they must add to 1: \n")
    numbers = prop_str.split()
    P = np.zeros(len(numbers))
    Psum = 0
    for i in range(len(numbers)):
        P[i] = float(numbers[i])
        Psum += P[i]
    if Psum != 1:
        print("Prior probabilities do not add up to 1. Please check!")
        sys.exit(1)
    
    # derive the decision rule from the training set and apply on the test set
    t0 = time.time()           # start time
    Y_pred = mpp(X_train, Y_train, X_test, cases, P)
    t1 = time.time()           # ending time

    print(Y_pred)
    Y_pred = Y_pred.astype("int")
    Y_pred = Y_pred.astype("str")
    # calculate accuracy
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
    print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore, 3), round(acs(Y_test, Y_pred), 3)))

    cm = confusion_matrix(Y_test, Y_pred)
    class_label = ["0", "1"]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


    print(f'The learning process takes {t1 - t0} seconds.')


if __name__ == "__main__":
    main()
