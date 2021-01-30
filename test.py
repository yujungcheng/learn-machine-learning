#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
#ts.plot()
#plt.show()



from pandas.plotting import scatter_matrix

df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
scatter_matrix(df)

import matplotlib.pyplot as plt

np.random.seed(134)
N = 1000

x1 = np.random.normal(0, 1, N)
x2 = x1 + np.random.normal(0, 3, N)
x3 = 2 * x1 - x2 + np.random.normal(0, 2, N)


df = pd.DataFrame({'x1':x1,
                   'x2':x2,
                   'x3':x3})

pd.plotting.scatter_matrix(df)

plt.show()
