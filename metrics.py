import numpy as np

def accuracy_score(y_true, y_pred):
    accuracy = (y_true==y_pred).sum() / len(y_true)
    return accuracy

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true,y_pred))