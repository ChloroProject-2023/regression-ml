import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

path = "./Results/"

class RFR:
    def __init__(self) -> None:
        self.description = "Random Forest Regression model: An ensemble method using a bundle of Decision Trees to determine the output based on them"
        self.model = RandomForestRegressor(n_estimators = 50, random_state = 42, bootstrap=True, min_samples_leaf=3)
        self.coef = None


    def __repr__(self):
        return self.description

    def train(self, X, y):
        self.model.fit(X, y)



    def metrics(self, X, y):
        metrics = dict()
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics['mse'] = mse
        metrics['r2'] = r2
        return metrics


    def inference(self, X):
        y_pred = self.model.predict(X)
        return y_pred

