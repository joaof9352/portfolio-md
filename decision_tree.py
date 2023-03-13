import numpy as np
from Dataset import Dataset

class Decision_tree:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _information_gain(self, X, y):
        for i in range(X.shape[1]):
            number_occurences = {}
            for j in range(X.shape[0]):
                if X[j, i] in number_occurences:
                    number_occurences[X[j, i]] += 1
                else:
                    number_occurences[X[j, i]] = 1
        
        total = number_occurences.items().apply(lambda x: x[1]).sum()
        probabilities = {k: v/total for k, v in number_occurences.items()}
        print(probabilities)

def information_gain(X, y):
        for i in range(X.shape[1]):
            number_occurences = {}
            for j in range(X.shape[0]):
                if X[j, i] in number_occurences:
                    number_occurences[X[j, i]] += 1
                else:
                    number_occurences[X[j, i]] = 1
        
        total = sum([x[1] for x in list(number_occurences.items())])
        probabilities = {k: v/total for k, v in number_occurences.items()}
        print(probabilities)

d = Dataset()
d.load(filename='notas.csv')
information_gain(d.X, d.y)

