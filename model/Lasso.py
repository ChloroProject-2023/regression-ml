import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

class Lasso_Regression:
    def __init__(self, alpha=1) -> None:
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha)
        self.coef = None
        self.intercept = None
    
    def fit(self, X):
        params = dict()
        self.model.fit(X)
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        params['intercept'] = self.intercept
        params['coef'] = self.coef
        return params
    
    def metrics(self, X, y):
        metrics = dict()
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics['mse'] = mse
        metrics['r2'] = r2
        return metrics
    
    def inference(self, X):
        n = X.shape[0]
        bias = np.ones((n, 1))
        X_new = np.concatenate((bias, X), axis=1)
        intercept = np.array([self.intercept])
        coef_new = np.concatenate((intercept, self.coef), axis=0)
        y = np.dot(X_new, coef_new.T)
        return y

