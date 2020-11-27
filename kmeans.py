import load_data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time



#load data
training_data = load_data.read_data("train.csv")
testing_data = load_data.read_data("test.csv")
testing_labels = load_data.read_data("submission.csv")
X_train, X_test = load_data.vectorize_data(training_data, testing_data)

Y_train = np.array(training_data)[:, -1]
Y_test = np.array(testing_labels)[:, -1]




#grid searching
clusters = list(range(1,30,1))
param_grid = {
    'n_clusters' : [2],
    'init' : ['k-means++', 'random'],
    'n_init': list(range(1,30,10)),
    'max_iter' : [1000]
}

clf = KMeans()
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 6)
grid_search.fit(X_train, Y_train)
print(grid_search.cv_results_)
print(grid_search.best_params_)
#Best Params: {}





mlp = KMeans(n_clusters=grid_search.best_params_['n_clusters'], n_init=grid_search.best_params_['n_init'], init=grid_search.best_params_['init'], max_iter=grid_search.best_params_['max_iter'])
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)

#typecast to string to match X_test format
Y_pred = [str(x) for x in Y_pred]

precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(Y_test, Y_pred), 3)))

cm = confusion_matrix(Y_test, Y_pred)
class_label = ["0", "1"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()