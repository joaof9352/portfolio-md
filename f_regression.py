import numpy as np
from scipy import stats
from typing import Tuple, Union
from sklearn.linear_model import LinearRegression
from Dataset import Dataset

def f_regression(dataset : Dataset) -> Union[Tuple[np.ndarray,np.ndarray],Tuple[float,float]]:

    X = dataset.X
    y = dataset.y
    n_features = X.shape[1]
    F = np.empty(n_features)
    p = np.empty(n_features)
    for i in range(n_features):
        lr = LinearRegression()
        lr.fit(X[:, i:i+1], y)
        y_pred = lr.predict(X[:, i:i+1])
        residual = y - y_pred
        degrees_of_freedom = len(y) - 2
        mse = np.sum(residual**2) / degrees_of_freedom
        ssx = np.sum((X[:, i] - np.mean(X[:, i]))**2)
        F[i] = ssx / mse
        p[i] = 1 - stats.f.cdf(F[i], 1, degrees_of_freedom)
    return F, p

