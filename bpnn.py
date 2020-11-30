import load_data
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time


#test various paramters using param_grid
def grid_search_bpnn(X_train, Y_train):
    #grid searching
    param_grid = {
        'max_iter' : [100000],
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'hidden_layer_sizes': [(500,100,50)],
        'learning_rate_init': [0.0009]
    }
    clf = MLPClassifier()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1)
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    return grid_search.best_params_

#bpnn with best results
def bpnn(X_train, Y_train, X_test, Y_test, params):
    start = time.time()

    mlp = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], activation=params['activation'], learning_rate_init=params['learning_rate_init'], solver=params['solver'], max_iter=100000)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)

    end = time.time()

    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
    print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore, 3), round(acs(Y_test, Y_pred), 3)))

    print("Execution time: " + str(end - start))

    cm = confusion_matrix(Y_test, Y_pred)
    class_label = ["0", "1"]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def main():

    # load data
    training_data = load_data.read_data("train.csv")
    testing_data = load_data.read_data("test.csv")
    testing_labels = load_data.read_data("submission.csv")
    X_train, X_test = load_data.vectorize_data(training_data, testing_data)

    Y_train = np.array(training_data)[:, -1]
    Y_test = np.array(testing_labels)[:, -1]

    #uncommment for grid searching
    #params = grid_search_bpnn(X_train, Y_train)

    params = {
        'activation': 'relu',
        'solver': 'lbfgs',
        'hidden_layer_sizes': (100,10),
        'learning_rate_init': 0.0009
    }

    bpnn(X_train, Y_train, X_test, Y_test, params)

if __name__ == "__main__":
    main()