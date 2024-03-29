import numpy as np
from Dataset import Dataset

class MLP:
    
    def __init__(self, dataset, hidden_nodes = 2, normalize = False):
        self.X, self.y = dataset.X, dataset.y
        self.X = np.hstack((np.ones([self.X.shape[0],1]), self.X))
        
        self.h = hidden_nodes
        self.W1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.W2 = np.zeros([1, hidden_nodes+1])
        
        if normalize:
            self.normalize()
        else:
            self.normalized = False


    def setWeights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2
        

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        if self.normalized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: x[1:] = (x[1:] - self.mu)

        
        z_2 = np.dot(self.W1,x)
        a_2 = np.empty([z_2.shape[0] + 1])
        a_2[0] = 1
        a_2[1:] = sigmoid(z_2)
        z_3 = np.dot(self.W2,a_2)
        
        return sigmoid(z_3)


    def costFunction(self):
        
        m = self.X.shape[0]
        Z2 = np.dot(self.X, self.W1.T)
        A2 = np.hstack ( (np.ones([Z2.shape[0],1]), sigmoid(Z2)))
        Z3 = np.dot(A2, self.W2.T)
        predictions = sigmoid(Z3)
        sqe = (predictions - self.y.reshape(m,1)) ** 2
        res = np.sum(sqe) / (2*m)
        return res

    def build_model(self):
        from scipy import optimize

        size = self.h * self.X.shape[1] + self.h+1
        
        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        weights = result.x
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def test():
    ds= Dataset()
    ds.load("numeros.csv")
    nn = MLP(ds, 2, normalize = False)
    nn.build_model()
    print(nn.predict(np.array([5,14,10,21,32,8,1])))
    print(nn.costFunction())
    
test()