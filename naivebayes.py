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
        # Obtém as classes únicas presentes nos rótulos
        self.classes = np.unique(y)
        # Inicializa os arrays de probabilidades a priori e de verossimilhança de classe
        self.class_priors = np.zeros(len(self.classes))
        self.class_likelihoods = []

        # Para cada classe, calcula a probabilidade a priori e a verossimilhança
        for i, c in enumerate(self.classes):
            # Filtra as amostras que pertencem à classe atual
            X_c = X[y == c]
            # Calcula a probabilidade a priori da classe
            self.class_priors[i] = X_c.shape[0] / X.shape[0]

            # Calcula a verossimilhança para cada atributo
            likelihoods = []
            for j in range(X.shape[1]):
                # Obtém os valores únicos do atributo
                feature_values = np.unique(X[:, j])
                # Inicializa o array de verossimilhança para o atributo atual
                likelihood = np.zeros(len(feature_values))
                # Calcula a frequência relativa de cada valor do atributo na classe atual
                for k, v in enumerate(feature_values):
                    likelihood[k] = np.sum(X_c[:, j] == v) / X_c.shape[0]
                likelihoods.append(likelihood)
            # Adiciona as verossimilhanças da classe atual à lista de verossimilhanças de classe
            self.class_likelihoods.append(likelihoods)


    def predict(self, X):
        predictions = []
        # Para cada amostra de teste
        for sample in X:
            posterior_probs = []
            # Para cada classe
            for i, c in enumerate(self.classes):
                # Calcula o logaritmo da probabilidade a priori da classe
                class_prior = np.log(self.class_priors[i])
                # Obtém as verossimilhanças da classe atual
                class_likelihoods = self.class_likelihoods[i]
                posterior_prob = class_prior
                # Para cada atributo da amostra
                for j in range(len(sample)):
                    feature_value = sample[j]
                    # Verifica se o valor do atributo está presente nos valores únicos do conjunto de treinamento
                    if feature_value in np.unique(X[:, j]):
                        # Obtém o índice do valor do atributo na lista de valores únicos
                        feature_index = np.where(np.unique(X[:, j]) == feature_value)[0][0]
                        # Obtém a verossimilhança do valor do atributo na classe atual
                        likelihood = class_likelihoods[j][feature_index]
                        # Adiciona o logaritmo da verossimilhança ao logaritmo da probabilidade a posteriori
                        posterior_prob += np.log(likelihood + 1e-9)  # Adiciona um valor pequeno para evitar o log(0)
                # Adiciona a probabilidade a posteriori à lista de probabilidades a posteriori
                posterior_probs.append(posterior_prob)
            # Obtém a classe com a maior probabilidade a posteriori como a predição
            prediction = self.classes[np.argmax(posterior_probs)]
            # Adiciona a predição à lista de predições
            predictions.append(prediction)
        # Retorna as predições como um array numpy
        return np.array(predictions)

    def calcAcc(self, X_test, y_test):
        # Calcula as predições para o conjunto de teste
        preds = self.predict(X_test)
        # Calcula a acurácia comparando as predições com os rótulos verdadeiros
        return accuracy_score(y_test, preds)
    
d = Dataset()
d.load(filename='notas.csv')
d.dropna()
n = NaiveBayes()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d.X, d.y, test_size=0.2, random_state=2023)

n.fit(X_train, y_train)
print(n.calcAcc(X_test, y_test))    