import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

# path = "./Results/"

class RFRegression:
    def __init__(self, dimension=3, min_samples_leaf = 3) -> None:
        self.name = "Random Forest Regression"
        self.dimesion = dimension  # dimension of the input data: pca 3, 5, or 7D, default is 3D
        self.description = "Random Forest Regression model: An ensemble method using a bundle of Decision Trees to determine the output based on them"
        
        self.model_N = RandomForestRegressor(n_estimators = 50, random_state = 42, bootstrap=True, min_samples_leaf=3)
        self.model_P = RandomForestRegressor(n_estimators = 50, random_state = 42, bootstrap=True, min_samples_leaf=3)
        self.model_K = RandomForestRegressor(n_estimators = 50, random_state = 42, bootstrap=True, min_samples_leaf=3)

        self.met = {}

        if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
            os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
        self.full_path = os.path.join(os.getcwd(), user, 'Result')
        self.train_data_path = 'merged.csv'
        self.split_ratio = 0.2

    def __repr__(self):
        return self.description

    def train(self, X, y):
        self.model_N.fit(X, y[:, 0])
        self.model_P.fit(X, y[:, 1])
        self.model_K.fit(X, y[:, 2])


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


    def inference(self, X):
        pass

    def predictNPK(self, X):
        
        N_pred = self.model_N.predict(X)
        P_pred = self.model_P.predict(X)
        K_pred = self.model_K.predict(X)
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
    def run(self, X_train, y_train, X_test, y_test, dimension):
        self.train(X_train, y_train)
        self.metrics(X_test, y_test)
        self.dimesion = dimension 
        fullpath = self.write_to_json(self.full_path)
        return fullpath

