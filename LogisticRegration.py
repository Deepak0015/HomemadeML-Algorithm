import numpy as np


def sigmoid(X):
    return (1/(1+np.exp - X))


class LogisticRegration():


    def __init__(self,lr=0.001,n_iter = 1000):
        self.lr  = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None 

    def fit(self,X,y):

        n_samples , n_features  = X.shape
        self.weights  = np.zeros(n_features)
        self.bias = 0

        for  _ in range(self.n_iter):
            liner_pre = np.dot(X,self.weights) + self.bias

            prediction  = sigmoid(liner_pre)

            dw = (1/n_samples) * np.dot(X.T,(prediction - y))
            db = (1/n_samples) * np.sum(prediction - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,X):
        liner_pre = np.dot(X,self.weights) + self.bias

        prediction  = sigmoid(liner_pre)



