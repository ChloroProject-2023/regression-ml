import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

class ExtremeGradBoost:
    def __init__(self, user):
        self.name = "Extreme Gradient Boosting"
        self.description = "Extreme Gradient Boosting model: "

        self.model_N = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                                        max_depth = 5, alpha = 10, n_estimators = 50)
        self.model_P = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                                        max_depth = 5, alpha = 10, n_estimators = 50)
        self.model_K = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                                        max_depth = 5, alpha = 10, n_estimators = 50)
        
        self.param = None
        self.met = {}

        if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
            os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
        self.full_path = os.path.join(os.getcwd(), user, 'Result')
        self.train_data_path = 'merged.csv'
        self.split_ratio = 0.2


    def __repr__(self) -> str:
        return self.description
    
    def train(self, X, y):
        pass