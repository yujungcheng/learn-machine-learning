#!/usr/bin/env python3

import numpy as np

# Native Bayes Calssifiers

# BernoulliNB
X = np.array([[0, 1, 0, 1],
             [1, 0, 1, 1],
             [0, 0, 0, 1],
             [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))

'''
class 0
[0, 1, 0, 1]
[0, 0, 0, 1]
=> 1st feature, zero 2 times, nonzero 0 times
=> 2nd feature, zero 1 times, nonzero 1 times
=> 3rd feature, zero 2 times, nonzero 0 times
=> 4th feature, zero 0 times, nonzero 2 times

class 1
[1, 0, 1, 1]
[1, 0, 1, 0]
=> 1st feature, zero 0 times, nonzero 2 times
=> 2nd feature, zero 2 times, nonzero 0 times
=> 3rd feature, zero 0 times, nonzero 2 times
=> 4th feature, zero 1 times, nonzero 1 times
'''
