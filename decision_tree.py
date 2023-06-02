import numpy as np
from Dataset import Dataset

class Decision_tree:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _entropy(self, X, y):
        for i in range(X.shape[1]):
            number_occurences = {}
            for j in range(X.shape[0]):
                if X[j, i] in number_occurences:
                    number_occurences[X[j, i]] += 1
                else:
                    number_occurences[X[j, i]] = 1
        
        total = number_occurences.items().apply(lambda x: x[1]).sum()
        probabilities = [v/total for _, v in number_occurences.items()]
        entropy = 0
        for probability in probabilities:
            entropy += probability * np.log2(probability)

        return -entropy


    def _choose_sub_classes(self):
        pass

d = Dataset()
d.load(filename='notas.csv')



