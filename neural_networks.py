#!/usr/bin/env python

import mglearn
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

print("\n[ Neural network model ]")
display(mglearn.plots.plot_logistic_regression_graph())
graphviz.Source(mglearn.plots.plot_logistic_regression_graph()).view()

display(mglearn.plots.plot_single_hidden_layer_graph())
graphviz.Source(mglearn.plots.plot_single_hidden_layer_graph()).view()
'''
Comment out second graphviz.Source() to show first one.
'''
#===============================================================================
# Hyperbolic tangent activation function and rectified linear activation function

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.show()

# Two hidden layer
graphviz.Source(mglearn.plots.plot_two_hidden_layer_graph()).view()

#===============================================================================
# Two moons dataset
print("\n[ Two moons dataset ]")

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#mlp = MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train) # not work
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
'''
Traceback (most recent call last):
  File "./neural_networks.py", line 39, in <module>
    mlp = MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train)
  File "/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/utils/validation.py", line 73, in inner_f
    return f(**kwargs)
TypeError: __init__() got an unexpected keyword argument 'algorithm'
Traceback (most recent call last):
  File "./neural_networks.py", line 40, in <module>
    mlp = MLPClassifier(solver='l-bfgs', random_state=0).fit(X_train, y_train)
  File "/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py", line 1027, in fit
    return self._fit(X, y, incremental=(self.warm_start and
  File "/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py", line 321, in _fit
    self._validate_hyperparameters()
  File "/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py", line 427, in _validate_hyperparameters
    raise ValueError("The solver %s is not supported. "
ValueError: The solver l-bfgs is not supported.  Expected one of: sgd, adam, lbfgs


Change "algorithm" to "solver", change "l-bfgs" to "lbftgs"
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
'''

# reduce hidden layer size to 10
print("\n- Reduce hidden layer size to 10")
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
'''
/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:471: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
'''

# using two hidden layers, with 10 units each, relu nonlinearity (default)
print("\n- Using two hidden layers, with 10 units each, relu nonlinearity (default)")
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


# using two hidden layers, with 10 units each, now with tanh nonlinearity
print("\n- Using two hidden layers, with 10 units each, now with tanh nonlinearity")
mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

#===============================================================================
# use l2 penalty
print("\n[ Use L2 penalty ]")

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
  for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
    mlp = MLPClassifier(solver='lbfgs',
                        random_state=0,
                        hidden_layer_sizes=[n_hidden_nodes,
                        n_hidden_nodes],
                        alpha=alpha)
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
  mlp = MLPClassifier(solver='lbfgs',
                      random_state=i,
                      hidden_layer_sizes=[100, 100])
  mlp.fit(X_train, y_train)
  mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
  mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
plt.show()

#===============================================================================
# Use MLPClassifier to Breast Cancer dataset
print("\n[ Use MLPClassifier to Breast Cancer dataset ]")

print("- Default paramaters")
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))


#### Rescale data
print("- Rescale data")

# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)

# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)

# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
'''
/home/ycheng/anaconda3/envs/machine-learning/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:582: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  warnings.warn(
'''

#### increase number of iterations
print("- Increase iterations")

mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

### introspect what weights were learned
print("\n Interospect learned weight")

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
