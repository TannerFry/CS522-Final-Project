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

#load data
training_data = load_data.read_data("train.csv")
testing_data = load_data.read_data("test.csv")
testing_labels = load_data.read_data("submission.csv")
X_train, X_test = load_data.vectorize_data(training_data, testing_data)

Y_train = np.array(training_data)[:, -1]
Y_test = np.array(testing_labels)[:, -1]


# Grid searching
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110, None],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
#rf = RandomForestClassifier()
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train, Y_train)
#print(grid_search.cv_results_)
#print(grid_search.best_params_)


print("Building Random Forest Model")
t0 = time.time()
rf = RandomForestClassifier(max_depth=None, n_estimators=150, n_jobs=-1)
rf_model = rf.fit(X_train, Y_train)
t1 = time.time()
print("Execution time: " + str(t1 - t0) + " seconds.")
Y_pred = rf.predict(X_test)

precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label='1', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(Y_test,Y_pred), 3)))

cm = confusion_matrix(Y_test, Y_pred)
class_label = ["0", "1"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()