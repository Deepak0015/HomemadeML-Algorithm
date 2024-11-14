import numpy as np

class SVM:
    def __init__(self,lr = 0.001,n_iters = 1000,lambda_param = 0.01):
       self.lr  = lr
       self.lambda_param = lambda_param
       self.n_iters  = n_iters
       self.weights = None
       self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]


    def predict(self,X):
        pred  = np.dot(X , self.weights) - self.bias 

        prediction  = np.sign(pred)
        return prediction


def accuracy(y_test,y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy
