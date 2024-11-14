import numpy as np

class PCA:
    def __init__(self , n_components):
        self.n_components = n_components
        self.components = None 
        self.mean =None 

    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        x_centered  = X - self.mean

        convariance_matrix  = np.cov(x_centered.T)
        eigenvalues , eigenvectors = np.linalg.eig(convariance_matrix)
        eigenvectors  = eigenvectors[: , np.argsort(-eigenvalues)]

        self.components = eigenvectors[: , :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered , self.components)
    


X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])
pca = PCA(n_components=1)
pca.fit(X)
X_reduced = pca.transform(X)
print('Reduced data:' , X_reduced)