import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RFR:
    def __init__(self) -> None:
        self.description = "Random Forest Regression model: An ensemble method using a bundle of Decision Trees to determine the output based on them"
        self.model = RandomForestRegressor(n_estimators = 10, random_state = 42, oob_score = True)
        self.coef = None


    def __repr__(self):
        return self.description

    def fit(self, X, y):
        pass


    def metrics(self, X, y):
        pass


    def inference(self, X):
        pass

