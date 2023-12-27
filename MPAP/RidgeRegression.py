import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

path = "./Results/"


class RidgeRegression:
    """
    Ridge Regression model:
    The formula is the same as Linear Regression model but the loss function is added
    with a regularization term to prevent overfitting and high variance.
    The loss function is: MSE + alpha * (sum of square of coefficients)
    alpha is a hyperparameter that controls the strength of regularization.

    params:
    alpha: float, default=1
        Strength of regularization

    methods:
    fit(X, y): fit the model with training data
    metrics(X, y): calculate metrics of the model (MSE, R2)
    inference(X): predict the output of the model with input X
    """


    def __init__(self, alpha=1) -> None:
        self.alpha = alpha 
        self.model = Ridge(alpha=self.alpha)
        self.name = "Ridge Regression"
        self.coef = None
        self.intercept = None
        self.description = "Ridge Regression model: The formula is the same as Linear Regression model but the loss function is added with a regularization term to prevent overfitting and high variance. The loss function is: MSE + alpha * (sum of square of coefficients) alpha is a hyperparameter that controls the strength of regularization."

    def __repr__(self):
        return self.description


    def train(self, X, y):
        params = dict()
        self.model.fit(X, y)
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        params['intercept'] = self.intercept
        params['coef'] = self.coef
        return params

    def metrics(self, X, y):
        met = dict()
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        met['mse'] = mse
        met['r2'] = r2
        return met

    def inference(self, X):
        # y = np.dot(X, self.coef.T) + self.intercept
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
    

    
