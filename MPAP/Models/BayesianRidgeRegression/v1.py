import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from typing import Dict, Any, List

path = "./Results/"
class BayesianRidgeRegression:
    def __init__(self) -> None:
        self.name: str = "BayesianRidge"
        # self.dimesion = dimension  # dimension of the input data: pca 3, 5, or 7D  

        self.model_N = BayesianRidge()
        self.model_P = BayesianRidge()
        self.model_K = BayesianRidge()  

        self.params: Dict[str, np.ndarray] = {}
        self.met: Dict[str, float] = {}
        self.description = "Bayesian Ridge Regression model: the same as Linear Regression but the loss function was added with the regularization terms which are the absolute summation of params of the hyperplane function"
        
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
            "N": self.params_N,
            "P": self.params_P,
            "K": self.params_K,
        }
        


    def join_params(self, coef, intercept):
        inter = np.array([intercept])
        p = np.concatenate((inter, coef))
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
        
        """
        Predicts the output based on the input features and given parameters.

        Args:
            X (np.ndarray): Input feature matrix.
            params (np.ndarray): Model parameters (including bias and coefficients).

        Returns:
            np.ndarray: Predicted values.
        """
        params = params.reshape(1, -1)
        X = np.atleast_2d(X).astype('float64')
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
        return {
            "N": N_pred[0][0],
            "P": P_pred[0][0],
            "K": K_pred[0][0]
        }

    def run(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray], X_test: np.ndarray, y_test: Dict[str, np.ndarray], dimension: int) -> None:
        """
        Executes the full process of training, evaluating, and saving the model.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (Dict[str, np.ndarray]): Training target values.
            X_test (np.ndarray): Testing feature matrix.
            y_test (Dict[str, np.ndarray]): Testing target values.
            dimension (int): Dimensionality of the PCA transformation.
            path (str): Filepath to save the model parameters and metrics.
        """
        self.train(X_train, y_train)
        self.metrics(X_test, y_test)
        self.dimension = dimension
        # Save model to JSON (implement this based on your requirements)


