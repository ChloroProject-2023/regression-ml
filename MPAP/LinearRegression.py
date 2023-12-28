import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

class Linear_Regression:
    def __init__(self, user) -> None:
        self.model = LinearRegression()
        self.name = "Linear Regression"
        self.coef = None
        self.intercept = None
        if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
            os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
        self.full_path = os.path.join(os.getcwd(), user, 'Result')
        self.train_data_path = 'merged.csv'
        self.split_ratio = 0.2

    def train(self, X, y):
        params = dict()
        self.model.fit(X, y)
        self.intercept = self.model.intercept_
        self.coef = self.model.coef_
        params['intercept'] = self.intercept
        params['coef'] = self.coef
        return params
    
    def __repr__(self):
        print("Test")
    
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
        intercept = np.array([self.intercept]).tolist()
        coef_new = np.concatenate((intercept, self.coef), axis=0)
        y = np.dot(X_new, coef_new.T)
        return y
    
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
    def run(self, X_train, y_train, X_test, y_test, path, dimension):
        self.params = self.train(X_train, y_train)
        self.met = self.metrics(X_test, y_test)
        self.dimesion = dimension 
        fullpath = self.write_to_json(self.full_path)
        return fullpath