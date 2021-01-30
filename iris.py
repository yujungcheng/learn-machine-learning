#!/usr/bin/env python


from sklearn.datasets import load_iris
iris_dataset = load_iris()

def print_line():
    print("\n"+ "="*80 + "\n")

#===============================================================================

print("keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'])
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names:\n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))

#===============================================================================

print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

#===============================================================================

import mglearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe,
                                 c=y_train,
                                 figsize=(10, 10),
                                 marker='0',
                                 hist_kwds={'bins': 20},
                                 s=60,
                                 alpha=.8,
                                 cmap=mglearn.cm3)
plt.show()


#===============================================================================

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn.fit(X_train, y_train))

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


#===============================================================================


y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#===============================================================================
