import load_data
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import time
import sklearn.metrics


def grid_search_knn(X_train, Y_train):
    #grid searching
    neighbors = list(range(1,50,1))

    param_grid = {
        'n_neighbors' : neighbors
    }

    clf = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1)
    grid_search.fit(X_train, Y_train)
    #print(grid_search.cv_results_)
    print(grid_search.best_params_)
    return grid_search.best_params_



def knn(X_train, Y_train, X_test, Y_test, params):

    start = time.time()

    mlp = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)

    end = time.time()

    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
    print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore, 3), round(acs(Y_test, Y_pred), 3)))

    print("Execution Time: " + str(end - start))

    cm = confusion_matrix(Y_test, Y_pred)
    class_label = ["0", "1"]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    sklearn.metrics.plot_roc_curve(mlp, X_test, Y_test)
    plt.title("ROC Curve")
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
    #params = grid_search_knn(X_train, Y_train)

    params = {
        'n_neighbors': 15
    }

    knn(X_train, Y_train, X_test, Y_test, params)






if __name__ == "__main__":
    main()