import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, X, y, normalize=False, regularization=False, lamda=1):
        self.X = X
        self.y = y
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        if normalize:
            self.normalize()
        else:
            self.normalized = False

    def fit(self, method='analytical', alpha=0.001, iterations=1000):
        if method == 'analytical':
            self.buildModel()
        elif method == 'gradient_descent':
            self.gradientDescent(alpha, iterations)
        else:
            raise ValueError("Invalid method. Choose 'analytical' or 'gradient_descent'.")

    def buildModel(self):
        if self.regularization:
            self.analyticalWithReg()
        else:
            self.theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

    def analyticalWithReg(self):
        matl = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(1, self.X.shape[1]):
            matl[i, i] = self.lamda
        mattemp = np.linalg.inv(self.X.T.dot(self.X) + matl)
        self.theta = mattemp.dot(self.X.T).dot(self.y)

    def predict(self, instance):
        x = np.hstack((np.ones((instance.shape[0], 1)), instance))
        print(x.shape)
        print(x)
        print(self.theta.transpose())
        if self.normalized:
            x[1:] = (x[1:] - self.mu) / self.sigma
        return np.dot(self.theta, x.transpose())

    def costFunction(self):
        m = self.X.shape[0]
        predictions = np.dot(self.X, self.theta)
        sqe = (predictions - self.y) ** 2
        res = np.sum(sqe) / (2 * m)
        return res

    def gradientDescent(self, alpha=0.001, iterations=1000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)
        if self.regularization:
            lamdas = np.zeros(self.X.shape[1])
            for i in range(1, self.X.shape[1]):
                lamdas[i] = self.lamda
        for its in range(iterations):
            J = self.costFunction()
            if its % 100 == 0:
                print(J)
            delta = self.X.T.dot(self.X.dot(self.theta) - self.y)
            if self.regularization:
                self.theta -= (alpha / m * (lamdas + delta))
            else:
                self.theta -= (alpha / m * delta)

    def printCoefs(self):
        print(self.theta)

    def plotData_2vars(self, xlab, ylab):
        plt.plot(self.X[:, 1], self.y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()

    def plotDataAndModel(self, xlab, ylab):
        plt.plot(self.X[:, 1], self.y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.plot(self.X[:, 1], np.dot(self.X, self.theta), '-')
        plt.legend(['Training data', 'Linear regression'])
        plt.show()

    def normalize(self):
        self.mu = np.mean(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] - self.mu
        self.sigma = np.std(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] / self.sigma
        self.normalized = True

d = Dataset()
d.load(filename='numeros.csv')

X_train, X_test, y_train, y_test = train_test_split(d.X, d.y, test_size=0.2, random_state=2023)

# Convert X_train and y_train to float data type
X_train = X_train.astype(float)
y_train = y_train.astype(float)

model = LinearRegression(X_train, y_train)
model.fit()
preds = model.predict(X_test)
print(preds)