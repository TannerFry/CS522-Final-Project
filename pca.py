import load_data
import numpy as np
import knn
import bpnn
from sklearn.decomposition import PCA

def pca(X_train, X_test,n_comp):
    pca = PCA(n_components=n_comp)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

def pca_error_rate(eigen_values, error_rate):
    dim = 0
    value_sum = np.sum(eigen_values)
    running_sum = 0
    for val in sorted(eigen_values):
        running_sum += val
        avg = running_sum / value_sum
        if avg > error_rate:
            break
        dim += 1

    return len(eigen_values) - dim

def main():
    # load data
    training_data = load_data.read_data("train.csv")
    testing_data = load_data.read_data("test.csv")
    testing_labels = load_data.read_data("submission.csv")
    X_train, X_test = load_data.vectorize_data(training_data, testing_data)

    Y_train = np.array(training_data)[:, -1]
    Y_test = np.array(testing_labels)[:, -1]

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.toarray()
    X_test = X_test.toarray()


    #
    # means = np.mean(X_train.T, axis=1)
    # # center columns
    # cols = X_train - means
    # # print(cols)
    #
    # # cov matrix
    # cov = np.cov(cols.T)
    #
    # # calculate dims needed to be kept based on error rate
    # values, vectors = np.linalg.eig(cov)
    # dim = pca_error_rate(values, 0.2)
    # print("Reduced DIMS to: " + str(dim) + " from " + str(len(training_data[0])))








    # reduce data
    X_train, X_test = pca(X_train, X_test, 2952)

    print(X_train.shape)
    print(X_test.shape)

    params = {
        'activation': 'relu',
        'solver': 'lbfgs',
        'hidden_layer_sizes': (100, 10),
        'learning_rate_init': 0.0009
    }

    bpnn.bpnn(X_train, Y_train, X_test, Y_test, params)




if __name__ == "__main__":
    main()