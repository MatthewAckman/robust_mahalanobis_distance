#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust multivariate outlier detection using Mahalanobis distance

NOTE: This project is incomplete

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
Mahanobis distance measures the distance between a point and a distribution
and can be thought of as multivariate euclidian distance. It is useful for 
detecting observations in a panel which are outliers in several dimensions.
'''


def mahalanobis(x=None, robust=True):
    
    '''
    Retuns an N*1 vector of mahalanobis distances for each row in a matrix.
    
    Params:
        mat {Pandas dataframe} - Input matrix, must have no NaNs
        robust {bool} - Default True, uses robust covariance matrix
    '''
    from sklearn.covariance import EmpiricalCovariance, MinCovDet

    assert len(X) == len(X.dropna()), 'Matrix must have no NaNs'

    cov = MinCovDet().fit(x) if robust else EmpiricalCovariance().fit(x)
    matrix_minus_mean = x - cov.location_
    inv_covariance_matrix = np.linalg.inv(cov.covariance_)
    md = np.dot(matrix_minus_mean, inv_covariance_matrix)
    
    return np.dot(md, matrix_minus_mean.T).diagonal()


'''
The practical utility of using robust Mahalanobis distance for outlier 
detection will be demonstrated by contaminating a generated 2D sample of
data points with a much sparser cloud of points. The naive (ie conventional 
covariance matrix) will fail to reject outliers as the contaminant cloud 
approaches the size or the original target cloud, whereas the robust MD
will be more successful.

Barring data visualization constraints, this process is scalable to n-dimenstions.
'''

## STATIC VARS

N_OBS = 5000  # number of observations
N_COL = 2  # number of columns
SH_OUT = 0.5  # share of observations which are outliers
N_OUT = int(SH_OUT * N_OBS)  # number of outliers
CMAP = 'viridis_r'  # matplotlib colourmap used for plots

## DEFINE COVARIANCES

cov = np.eye(N_COL)  # indentity matrix (N_OBS * N_COL)
cov[0,0] = 3  # manually set correlations

covC = np.eye(N_COL)  # covariance of contaminated sample
covC[np.arange(1,N_COL), np.arange(1,N_COL)] = 10  # stdev of contaminated obs

# CALCULATE AND DISPLAY MAHALANOBIS DISTANCE

X = np.dot(np.random.randn(N_OBS, N_COL), cov)

X[-N_OUT:] = np.dot(np.random.randn(N_OUT, N_COL), covC)  # contaminate sample

X = pd.DataFrame(X, columns=['Var'+str(x) for x in range(1, N_COL+1)])

X['cont'] = (X.index >= (1-SH_OUT)*N_OBS).astype(int)

X['NMD'] = mahalanobis(X[['Var1','Var2']], robust=False)  # naive mah. distance
X['RMD'] = mahalanobis(X[['Var1','Var2']], robust=True)  # robust mah. distance
X['pVal'] = 1 - chi2.cdf(X.RMD, df=N_COL)

fig, ax = plt.subplots(1,3,figsize=(14,6))

X.plot.scatter('Var1','Var2', c='cont', cmap=CMAP, ax=ax[0],
title='Originated vs contaminated observations', xlabel='', ylabel='')

X.plot.scatter('Var1', 'Var2', c='NMD', cmap=CMAP, ax=ax[1],
title='Non-robust mahalanobis distance',
xlabel='',ylabel='', vmin=0, vmax=20)

X.plot.scatter('Var1', 'Var2', c='RMD', cmap=CMAP, ax=ax[2],
title='Robust mahalanobis distance',
xlabel='',ylabel='', vmin=0, vmax=20)

fig.tight_layout()

fig.savefig('example.svg')