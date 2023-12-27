import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

path = "./Results/"

class Lasso_Regression:
    def __init__(self, alpha=1, dimension=3) -> None:
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha)
        self.coef = None
        self.intercept = None
        self.met = {}
        self.description = "Lasso Regression model: the same as Linear Regression but the loss function was added with the regularization terms which are the absolute summation of params of the hyperplane function"
        self.name = "Lasso"
        self.dimension = dimension # dimension of the input data: pca 3, 5, or 7D, default is 3D


    def train(self, X):
        params = dict()
        self.model.fit(X)
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        params['intercept'] = self.intercept
        params['coef'] = self.coef
        return params
    
    def __repr__(self):
        return self.description


    def metrics(self, X, y):
        
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        self.met['mse'] = mse
        self.met['r2'] = r2
        return self.met
    
    def inference(self, X):
        n = X.shape[0]
        bias = np.ones((n, 1))
        X_new = np.concatenate((bias, X), axis=1)
        intercept = np.array([self.intercept])
        coef_new = np.concatenate((intercept, self.coef), axis=0)
        y = np.dot(X_new, coef_new.T)
        return y
    
    def write_to_json(self, path):
        filename = self.name + str(self.dimension) + '.json'
        fullpath = os.path.join(path, filename)
        dict = {
            "name_model": self.name,
            "pca_dimension": self.dimension,
            "metrics": self.met,
            "params": self.params,
            "filepath": fullpath
        }
        with open(fullpath, 'w') as f:
            json.dump(dict, f)
            

