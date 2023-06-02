import numpy as np
import math
from metrics import accuracy_score
from Dataset import Dataset

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_likelihoods = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = np.zeros(len(self.classes))
        self.class_likelihoods = []

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / X.shape[0]

            likelihoods = []
            for j in range(X.shape[1]):
                feature_values = np.unique(X[:, j])
                likelihood = np.zeros(len(feature_values))
                for k, v in enumerate(feature_values):
                    likelihood[k] = np.sum(X_c[:, j] == v) / X_c.shape[0]
                likelihoods.append(likelihood)
            self.class_likelihoods.append(likelihoods)


    def predict(self, X):
        predictions = []
        for sample in X:
            posterior_probs = []
            for i, c in enumerate(self.classes):
                class_prior = np.log(self.class_priors[i])
                class_likelihoods = self.class_likelihoods[i]
                posterior_prob = class_prior
                for j in range(len(sample)):
                    feature_value = sample[j]
                    if feature_value in np.unique(X[:, j]):
                        feature_index = np.where(np.unique(X[:, j]) == feature_value)[0][0]
                        likelihood = class_likelihoods[j][feature_index]
                        posterior_prob += np.log(likelihood + 1e-9)  # Adiciona um valor pequeno para evitar o log(0)
                posterior_probs.append(posterior_prob)
            prediction = self.classes[np.argmax(posterior_probs)]
            predictions.append(prediction)
        return np.array(predictions)

    def calcAcc(self, X_test, y_test):
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)
    
d = Dataset()
d.load(filename='numeros.csv')
d.dropna()
n = NaiveBayes()

X_train, X_test, y_train, y_test = d.train_test_split(test_size=0.2, random_state=2023)

n.fit(X_train, y_train)
print(n.calcAcc(X_test, y_test))    