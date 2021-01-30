#!/usr/bin/env python
# K Nearest Neighbors


import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


#===============================================================================

# binary dataset
#X, y = mglearn.datasets.make_forge()
X,y = make_blobs()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()



# wave dataset
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()



# breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))



# boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))



# k-NN algorithm
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
plt.close()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()
plt.close()

# FutureWarning issue
'''
When run with following warning message.
/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/utils/deprecation.py:86: FutureWarning: Function make_blobs is deprecated; Please import make_blobs directly from scikit-learn
  warnings.warn(msg, category=FutureWarning)

It is mglearn's bug, you can fix it by edit /anaconda3/envs/machine-learning/lib/python3.8/site-packages/mglearn/datasets.py

#from .make_blobs import make_blobs
from sklearn.datasets import make_blobs

Reference:
https://github.com/amueller/introduction_to_ml_with_python/issues/113
https://github.com/amueller/introduction_to_ml_with_python/commit/b54511da61b4283e415951d92feba70100914324
'''


#===============================================================================

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# Decision boundary
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for n_neighbors, ax in zip([1, 3, 9, 12], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
plt.close()


''' Numpy slice assignment
[0]     #means line 0 of your matrix
[(0,0)] #means cell at 0,0 of your matrix
[0:1]   #means lines 0 to 1 excluded of your matrix
[:1]    #excluding the first value means all lines until line 1 excluded
[1:]    #excluding the last param mean all lines starting form line 1
         included
[:]     #excluding both means all lines
[::2]   #the addition of a second ':' is the sampling. (1 item every 2)
[::]    #exluding it means a sampling of 1
[:,:]   #simply uses a tuple (a single , represents an empty tuple) instead
         of an index.

>>> a = numpy.arange(20).reshape(4,5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
>>> a[0]
array([0, 1, 2, 3, 4])
>>> a[(0,0)]
0
>>> a[:1]
array([[0, 1, 2, 3, 4]]
>>> a[:, 0]
array([ 0,  5, 10, 15])
>>> a[:, 1]
array([ 1,  6, 11, 16])
>>> a[:, 4]
array([ 4,  9, 14, 19])
>>> a[1:]
array([[ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
>>> a[:,:]
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
>>> a[::2]
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14]])

>>> a[::3]
array([[ 0,  1,  2,  3,  4],
       [15, 16, 17, 18, 19]])

'''


#===============================================================================
# Use breast cancer data to test decision boundary
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
plt.close()



#===============================================================================
# KNN Regression
# Explation https://www.saedsayad.com/k_nearest_neighbors_reg.htm

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("Test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))



#===============================================================================
fig, axes = plt.subplots(1, 3,  figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(2), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors,
                                                                                  reg.score(X_train, y_train),
                                                                                  reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
plt.show()


#===============================================================================
