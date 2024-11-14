import numpy as np

class SimpleGradientBoosting:
    def __init__(self , n_estimtors = 10 , learning_rate = 0.1):
        self.n_estimators = n_estimtors
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def _fit_stump(self, X , residual):
        thresholds = np.unique(X)
        best_threshold , best_mse = None , float('inf')

        for threshold  in thresholds:
            pred = np.where(X>threshold , 1 , -1)
            mse = np.mean((residual - pred )**2)
            if mse < best_mse:
                best_mse = mse 
                best_threshold = threshold


        return  best_threshold , best_mse
    

    def fit(self  , X , y ):
        residual = y.astype(float)
        for _ in range(self.n_estimators):
            threshold , _ = self._fit_stump(X , residual=residual)
            pred = np.where(X >threshold , 1 , -1)
            residual -= self.learning_rate * pred 
            self.models.append(threshold)
            self.model_weights.append(self.learning_rate)


    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for threshold , weight  in zip(self.models  , self.model_weights):
            preds += weight * np.where(X>threshold , -1 , -1)

        return np.sign(preds)
    
X = np.array([1,2,3,4,5,6,7])
y = np.array([-1, -1, -1, 1,1,1,1])
gb = SimpleGradientBoosting(n_estimtors=5, learning_rate=0.1)
gb.fit(X ,y)
predictions = gb.predict(X)
print("Predictions :" , predictions)
