import numpy as np
import math
from sklearn.metrics import accuracy_score
from Dataset import Dataset

class NaiveBayes(object):

    def __init__(self, alpha=1.0):
        super(NaiveBayes).__init__()
        self.prior = None
        self.lk = None
        self.alpha = alpha

    def fit(self, dataset: Dataset):
        X,y = dataset.X, dataset.y
        self.dataset = Dataset
        r = X.shape[0]

        XpC = np.array([X[y==i] for i in np.unique(y)], dtype=object)
        self.prior = np.array([len(Xc) / r for Xc in XpC])
        total = np.array([res.sum(axis=0) for res in XpC]) + self.alpha
        self.lk = total / total.sum(axis=1).reshape(-1,1)
        self.checkFit = True

    def predictProb(self, n):
        assert self.checkFit, 'Fit the model first'

        enumerateClass = np.zeros(shape=(n.shape[0], self.prior.shape[0]))
        for i, n in enumerate(n):
            sig = n.astype(bool)
            lkCurrent = self.lk[:,sig] ** n[sig]
            lkMarginal = lkCurrent.prod(axis=1)
            enumerateClass[i] = lkMarginal * self.prior

        normalize = enumerateClass.sum(axis=1).reshape(-1,1)
        probCond = enumerateClass / normalize
        assert (probCond.sum(axis=1) - 1 < 0.001).all(), 'the rows sum must be 1'
        return probCond

    def predict(self, n):

        assert self.checkFit, 'Fit the model first'
        return self.predictProb(n).argmax(axis=1)

    def calcAcc(self, X = None, y = None):

        assert self.checkFit, 'Fit the model first'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = self.predict(X)

        return accuracy_score(y,y_pred)