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


    def __init__(self, user, alpha=1) -> None:
        self.name = "Ridge Regression"
        self.description = "Ridge Regression model: The formula is the same as Linear Regression model but the loss function is added with a regularization term to prevent overfitting and high variance. The loss function is: MSE + alpha * (sum of square of coefficients) alpha is a hyperparameter that controls the strength of regularization."
        # self.dimesion = dimension  # dimension of the input data: pca 3, 5, or 7D, default is 3D

        self.model_N = Ridge(alpha=alpha)
        self.model_P = Ridge(alpha=alpha)
        self.model_K = Ridge(alpha=alpha)

        self.params = {}
        self.met = {}

        if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
            os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
        self.full_path = os.path.join(os.getcwd(), user, 'Result')
        self.train_data_path = 'merged.csv'
        self.split_ratio = 0.2


    def __repr__(self):
        return self.description


    def train(self, X, y):
        self.model_N.fit(X, y['N conc. (mg/kg)'])
        self.params_N = self.join_params(self.model_N.coef_, self.model_N.intercept_)

        self.model_P.fit(X, y['P conc. (mg/kg)'])
        self.params_P = self.join_params(self.model_P.coef_, self.model_P.intercept_)

        self.model_K.fit(X, y['K conc. (mg/kg)'])
        self.params_K = self.join_params(self.model_K.coef_, self.model_K.intercept_)

        self.params = {
            "params_N": self.params_N,
            "params_P": self.params_P,
            "params_K": self.params_K,
        }
        
    def join_params(self, coef, intercept):
        inter = np.array([intercept])
        p = np.concatenate((inter, coef), axis=1)
        return p

    def metrics(self, X, y):
        y_pred_N = self.model_N.predict(X)
        y_pred_P = self.model_P.predict(X)
        y_pred_K = self.model_K.predict(X)

        mse_N = mean_squared_error(y['N conc. (mg/kg)'], y_pred_N)
        mse_P = mean_squared_error(y['P conc. (mg/kg)'], y_pred_P)
        mse_K = mean_squared_error(y['K conc. (mg/kg)'], y_pred_K)
        average_mse = (mse_N + mse_P + mse_K) / 3

        r2_N = r2_score(y['N conc. (mg/kg)'], y_pred_N)
        r2_P = r2_score(y['P conc. (mg/kg)'], y_pred_P)
        r2_K = r2_score(y['K conc. (mg/kg)'], y_pred_K)
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

        y = np.dot(X_new, coef_new.T)
        return y
    

    def predictNPK(self, X):
        N_pred = self.inference(X, self.params_N)
        P_pred = self.inference(X, self.params_P)
        K_pred = self.inference(X, self.params_K)
        return N_pred, P_pred, K_pred


    def write_to_json(self, path):
        # convert all attributes to list
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                self.params[key] = value.tolist()
            else:
                self.params[key] = value
        for key, value in self.met.items():
            if isinstance(value, np.ndarray):
                self.met[key] = value.tolist()
            else:
                self.met[key] = value
                
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                self.__dict__[key] = value.tolist()
            else:
                self.__dict__[key] = value
                
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
            json.dump(dict, f, indent=4)
        return fullpath.replace('\\', '/')

    # This function will automatically run and save the model to the path
    def run(self, X_train, y_train, X_test, y_test, dimension=3):
        self.train(X_train, y_train)
        self.metrics(X_test, y_test)
        self.dimesion = dimension 
        fullpath = self.write_to_json(self.full_path)
        return fullpath
    

    
