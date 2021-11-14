#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust multivariate outlier detection using Mahalanobis distance
@author: mattackman
"""

## STANDARD
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')

## STATS & MODELLING
import statsmodels.api as sm
import scipy as sp
from scipy.stats import chi2

## SUPRESS SETTINGWITHCOPY WARnING
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore")

'''
Mahanobis distance measures the distane between a point and a distribution
and can be thought of as multivariate euclidian distance. It is useful for 
detecting observations in a panel which are outliers in several dimensions.
'''


def mahalanobis(x=None, robust=True):
    
    '''
    Retuns an N*1 vector of mahalanobis distances for each obs in a matrix
    
    Params:
        mat {Pandas dataframe} - Input matrix, must have no NaNs
        robust {bool} - Default True, uses robust covariance matrix
    '''
    from sklearn.covariance import EmpiricalCovariance, MinCovDet

    cov = MinCovDet().fit(x) if robust else EmpiricalCovariance().fit(x)
    matrix_minus_mean = x - cov.location_
    inv_covariance_matrix = np.linalg.inv(cov.covariance_)
    md = np.dot(matrix_minus_mean, inv_covariance_matrix)
    
    return np.sqrt(np.dot(md, matrix_minus_mean.T).diagonal())


'''
The practical utility of using robust Mahalanobis distance for outlier 
detection will be demonstrated by contaminating a generated 2D sample of
data points with a much sparser cloud of points. The naive (ie conventional 
ovariance matrix) will fail to reject outliers as the contaminant cloud 
approaches the size or the original target cloud, whereas the robust MD
will be more successful.
'''

## STATIC VARS

nObs = 5000  # number of observations
nCol = 2  # number of columns
shOut = 0.1  # share of observations which are outliers
nOut = int(shOut * nObs)

cov = np.eye(nCol)  # indentity matrix (nCol * nCol)
cov[0,0] = 3  # covariance of general obs
cov[0,1] = 2  # covariance of general obs


covC = np.eye(nCol)  # covariance of contaminated sample
covC[np.arange(1,nCol), np.arange(1,nCol)] = 3  # stdev of contaminated obs

# RMD

X = np.dot(np.random.randn(nObs, nCol), cov)


X[-nOut:] = np.dot(np.random.randn(nOut, nCol), covC)

X = pd.DataFrame(X, columns=['Var1','Var2'])

X['cont'] = (X.index >= (1-shOut)*nObs).astype(int)

X['RMD'] = mahalanobis(X[['Var1','Var2']], robust=True)
X['Pval'] = np.sqrt(chi2.ppf(0.95, df=nCol))

fig, ax = plt.subplots(1,2)

X.plot.scatter('Var1','Var2', c='cont', cmap='viridis', ax=ax[0])
X.plot.scatter('Var1', 'Var2', c='Pval', cmap='viridis', ax=ax[1])

fig.tight_layout()

# %%
