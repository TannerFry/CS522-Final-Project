import load_data
import numpy as np
from sklearn.decomposition import PCA

def pca(X_train, X_test,n_comp):
    pca = PCA(n_components=n_comp)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

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

    # reduce data
    X_train, X_test = pca(X_train, X_test, 5)

    print(X_train.shape)
    print(X_test.shape)

if __name__ == "__main__":
    main()