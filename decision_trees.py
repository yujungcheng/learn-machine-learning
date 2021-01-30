#!/usr/bin/env python3

import mglearn
import matplotlib.pyplot as plt


mglearn.plots.plot_animal_tree()
plt.show()

'''
conda install graphviz python-graphviz

https://github.com/amueller/introduction_to_ml_with_python/issues/69

(machine-learning) ycheng@nuc:~/machine-learning$ ./decision_trees.py
Traceback (most recent call last):
  File "./decision_trees.py", line 12, in <module>
    mglearn.plots.plot_animal_tree()
  File "/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/mglearn/plot_animal_tree.py", line 6, in plot_animal_tree
    import graphviz
ModuleNotFoundError: No module named 'graphviz'

'''

#===============================================================================
# Decision tree in Breast cancel data set

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("[ without pre-pruning (Unpruned tree) ]")
print("- Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("- Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# with pre-pruning
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("[ with pre-pruning ]")
print("- Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("- Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# Analyzing decision trees
from sklearn.tree import export_graphviz
export_graphviz(tree,
                out_file="tree.dot",
                class_names=["maligant", "benign"],
                feature_names=cancer.feature_names,
                impurity=False,
                filled=True)
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).view()

# "Feature importance in trees"
print("- Feature importances:\n{}".format(tree.feature_importances_))

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
plot_feature_importances_cancer(tree)


#===============================================================================
# Two-dimensional data set
import matplotlib.pyplot as plt
from IPython.display import display

print("[ Two-dimensional data set ]")
tree = mglearn.plots.plot_tree_not_monotone()
display(tree) # does not display image
plt.show() # can show image
plt.close()

#===============================================================================
# DecisionTreeRegressor to ram price data set
'''
download ram_price.csv
https://github.com/amueller/introduction_to_ml_with_python/blob/master/data/ram_price.csv
'''

#### show data
import pandas as pd
ram_prices = pd.read_csv("data/ram_price.csv")
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
plt.show()


#### Comparison of predictions by linear model and DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# predict prices based on date
X_train = data_train.date[:, np.newaxis]

# we use a log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()
plt.show()
plt.close()
'''
The goal in this example is not to point time based data to the decision tree is not good model.
It is to explain that decision tree has special characteristic in its prediction method.
'''

#===============================================================================
# Ensembles of Decision Trees

# Two major and effective model:
# - Random forests
# - Gradient boosted regression trees (gradient boosting machines)

# Random forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()
plt.close()


#===============================================================================
# Ensembles of Decision Trees on Breast cancer data set

X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0) # set paramater "n_jobs=<integer>" to run multiple jobs in parallel.
forest.fit(X_train, y_train)
print("[ Random forest on Breast cancer data set ]")
print("- Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("- Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
plot_feature_importances_cancer(forest)


#===============================================================================
# Gradient boosted regression trees (gradient boosting machines)

from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("[ Gradient boosted regression trees on Breast cancer data set ]")
print("- Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("- Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# max_depth 1
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("[ Gradient boosted regression trees on Breast cancer data set, max_depth=1 ]")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# learning_rate 0.01
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("[ Gradient boosted regression trees on Breast cancer data set, learning_rate=0.01 ]")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# feature importances
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
plot_feature_importances_cancer(gbrt)
'''
This feature importance is quite differ to original in book.
'''
