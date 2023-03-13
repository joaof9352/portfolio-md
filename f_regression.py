import numpy as np
import scipy as stats
from typing import Tuple, Union
from Dataset import Dataset

def f_regression(dataset : Dataset) -> Union[Tuple[np.ndarray,np.ndarray],Tuple[float,float]]:

    X = dataset.X
    y = dataset.y
    coef = np.array([stats.pearsonr(X[:,i], y)[0] for i in range(X.shape[1])])
    freedom_deg = y.size - 2
    coef_sqr = coef ** 2
    F = coef_sqr / (1 - coef_sqr) * freedom_deg
    p = stats.f.sf(F, 1, freedom_deg)

    return F,p

