import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

path = "./Results/"
class BayesianRidgeRegression:
    def __init__(self, dimension) -> None:
        self.model = BayesianRidge()
        self.name = "BayesianRidge"
        self.dimesion = dimension  # dimension of the input data: pca 3, 5, or 7D    
        self.coef = None
        self.intercept = None
        self.params = {}
        self.met = {}
        self.description = "Bayesian Ridge Regression model: the same as Linear Regression but the loss function was added with the regularization terms which are the absolute summation of params of the hyperplane function"
        # self.path = "./Results/"

    def __repr__(self):
        return self.description

    def train(self, X, y):
        self.model.fit(X, y)
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        self.params['intercept'] = self.intercept
        self.params['coef'] = self.coef
        return self.params

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
        filename = self.name + str(self.dimesion) + '.json'
        fullpath = os.path.join(path, filename)
        dict = {
            "name_model": self.name,
            "pca_dimension": self.dimesion,
            "metrics": self.met,
            "params": self.params,
            "filepath": fullpath
        }
        with open(fullpath, 'w') as f:
            json.dump(dict, f)
            


# dict = {
#     "name": "BayesianRidge",
#     "dimension": 3,
# }

# name = "River"

# path = "./Results/"
# filename = "BayesianRidge.json"

# def test(self, path, filename):
#     fullpath = os.path.join(path, filename)

#     with open(fullpath, 'w') as f:
#             json.dump(name, f)
#             json.dump(dict, f)


# if __name__ == "__main__":
#     test(path, filename)
