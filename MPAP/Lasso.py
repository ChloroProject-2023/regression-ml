import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

path = "./Results/"

class Lasso_Regression:
    def __init__(self, alpha=1, dimension=3) -> None:
        self.alpha = alpha
        self.model_N = Lasso(alpha=self.alpha)
        self.model_P = Lasso(alpha=self.alpha)
        self.model_K = Lasso(alpha=self.alpha)
        self.params = {}
        self.met = {}
        self.description = "Lasso Regression model: the same as Linear Regression but the loss function was added with the regularization terms which are the absolute summation of params of the hyperplane function"
        self.name = "Lasso"
        self.dimension = dimension # dimension of the input data: pca 3, 5, or 7D, default is 3D

    def __repr__(self):
            return self.description
    

    def train(self, X, y):
        self.model_N.fit(X, y[:, 0])
        self.params_N = self.join_params(self.model_N.coef_, self.model_N.intercept_)

        self.model_P.fit(X, y[:, 1])
        self.params_P = self.join_params(self.model_P.coef_, self.model_P.intercept_)

        self.model_K.fit(X, y[:, 2])
        self.params_K = self.join_params(self.model_K.coef_, self.model_K.intercept_)

        self.params = {
            "params_N": self.params_N,
            "params_P": self.params_P,
            "params_K": self.params_K,
        }

    def join_params(self, coef, intercept):
        p = np.concatenate((intercept, coef), axis=1)
        return p
    

    def metrics(self, X, y):
        y_pred_N = self.model_N.predict(X)
        y_pred_P = self.model_P.predict(X)
        y_pred_K = self.model_K.predict(X)

        mse_N = mean_squared_error(y[:, 0], y_pred_N)
        mse_P = mean_squared_error(y[:, 1], y_pred_P)
        mse_K = mean_squared_error(y[:, 2], y_pred_K)
        average_mse = (mse_N + mse_P + mse_K) / 3

        r2_N = r2_score(y[:, 0], y_pred_N)
        r2_P = r2_score(y[:, 1], y_pred_P)
        r2_K = r2_score(y[:, 2], y_pred_K)
        average_r2 = (r2_N + r2_P + r2_K) / 3
        
        self.met['mse'] = average_mse
        self.met['r2'] = average_r2
        return self.met
    
    def inference(self, X, params):         # params: a 2D array of shape (1, N) with N = no_features + 1
        n = X.shape[0]
        bias = np.ones((n, 1))
        X_new = np.concatenate((bias, X), axis=1)

        coef_new = []
        for i in range(params.shape[1]):
            coef_new.append(params[0][i])

        coef_new = np.array([coef_new])

        # intercept = np.array([self.intercept])
        # coef_new = np.concatenate((intercept, self.coef), axis=0)
        y = np.dot(X_new, coef_new.T)
        return y
    

    def predictNPK(self, X):
        N_pred = self.inference(X, self.params_N)
        P_pred = self.inference(X, self.params_P)
        K_pred = self.inference(X, self.params_K)
        return N_pred, P_pred, K_pred 

    
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
            

