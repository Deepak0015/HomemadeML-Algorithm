import numpy as np

class LinearRegression:
    def __init__(self , learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations  = iterations
        self.weights = None 
        self.bias = None 

    def fit(self, X , y ):
        n_samples  , n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range( self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias 
            dw = (1 / n_samples)  * np.dot(X.T  , (y_pred - y ))
            db = (1 / n_samples) * np.sum(y_pred -  y)

            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    



X = np.array([[1] , [2],[3], [4], [5]])
y = np.array([3, 4, 7, 9, 11])
regressor = LinearRegression()
regressor.fit(X, y )

predictions =  regressor.predict(X)
print("Predictions:" , predictions)
