import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from typing import Dict, Any, List

class Linear_Regression:
    """
    Linear Regression model class which predicts and analyses N, P, and K concentrations.

    Attributes:
        name (str): Name of the model.
        description (str): Description of the model.
        model_N (LinearRegression): Linear regression model for N concentration.
        model_P (LinearRegression): Linear regression model for P concentration.
        model_K (LinearRegression): Linear regression model for K concentration.
        params (Dict[str, np.ndarray]): Parameters of the models.
        met (Dict[str, float]): Metrics of the models.
        train_data_path (str): Path to the training data.
        split_ratio (float): Ratio for splitting the dataset into training and testing.
    """

    def __init__(self) -> None:
        self.name: str = "Linear Regression"
        self.description: str = "Linear Regression model: a model that fits the data with a hyperplane function"
        self.model_N: LinearRegression = LinearRegression()
        self.model_P: LinearRegression = LinearRegression()
        self.model_K: LinearRegression = LinearRegression()
        self.params: Dict[str, np.ndarray] = {}
        self.met: Dict[str, float] = {}
        self.train_data_path: str = 'merged.csv'
        self.split_ratio: float = 0.2

    def __repr__(self) -> str:
        return self.description

    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> None:
        """
        Trains the Linear Regression models using the provided dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (Dict[str, np.ndarray]): Target values for N, P, and K concentrations.
        """

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

    def join_params(self, coef: np.ndarray, intercept: float) -> np.ndarray:
        """
        Combines coefficients and intercept into a single array.

        Args:
            coef (np.ndarray): Coefficients of the model.
            intercept (float): Intercept of the model.

        Returns:
            np.ndarray: Combined array of intercept and coefficients.
        """
        inter = np.array([intercept])
        return np.concatenate((inter, coef))

    def metrics(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculates and returns the performance metrics of the model.

        Args:
            X (np.ndarray): Feature matrix for testing.
            y (Dict[str, np.ndarray]): Actual target values for testing.

        Returns:
            Dict[str, float]: Dictionary containing MSE and R-squared metrics.
        """
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
        
        self.met = {'mse': average_mse, 'r2': average_r2}
        return self.met

    def inference(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
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
        params = np.array([params])
        for i in range(params.shape[1]):
            coef_new.append(params[0][i])

        coef_new = np.array([coef_new])

        return np.dot(X_new, coef_new.T)

    def predictNPK(self, X: np.ndarray) -> Dict[str, float]:
        """
        Predicts N, P, and K concentrations based on the input features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            Dict[str, float]: Predicted N, P, and K concentrations.
        """
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

