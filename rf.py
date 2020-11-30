import load_data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

def rf(X_train, Y_train, X_test, Y_test):

    start = time.time()

    rf = RandomForestClassifier(max_depth=None, n_jobs=-1)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    end = time.time()

    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
    print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore, 3), round(acs(Y_test, Y_pred), 3)))
    print("Execution time: " + str(end - start) + " seconds.")

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

    rf(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()