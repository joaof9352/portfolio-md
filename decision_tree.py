import numpy as np

class Decision_tree:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _information_gain(self, X, y):
        for i in range(X.shape[1]):
            number_occurences = len(np.unique(X[:, i]))