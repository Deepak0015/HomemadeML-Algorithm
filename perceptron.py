import numpy as np 


def unit_step_func(x):
    return np.where( x >0 , 1 ,0)




class Percpetron:

    def __init__(self,learning_rate = 0.001,n_iter = 1000):

        self.lr  = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.activation_function = unit_step_func


    def fit(self,X,y):
        n_samples , n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.where( y > 0 , 1,0)
        #learn weights 
        for _ in range(self.n_iter):
            for idx , x_i in enumerate(X):
                linear_predict = np.dot(x_i , self.weights) + self.bias
                y_predict = self.activation_function(linear_predict)

                #update the weights and bias in preceptron update rule

                update = self.lr *(y[idx] - y_predict)

                self.weights += update * x_i
                self.bias  += update


    def predict(self,X):
        linear_predict = np.dot(X,self.weights) + self.bias
        prediction = self.activation_function(linear_predict)
        return prediction
    

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy