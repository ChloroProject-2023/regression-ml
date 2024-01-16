import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import importlib
import configparser
import re
from utils.write_json import write_to_json 

class ReflectenceService:
    def __init__(self, user, model_name, ndim, version):
        self.user = user
        self.model_name = model_name
        self.cfg = configparser.ConfigParser()
        self.cfg.read('config.ini')
        self.base_dir = self.cfg['DEFAULT']['BASE_DIR']
        self.prepare_base_dir()
        self.version = version
        self.data = self.get_csv(self.user, self.model_name, self.version)

    def get_csv(self, user, model_name, version):
        import_path = f"{user}.Models.{model_name}.{version}"
        module = importlib.import_module(import_path)
        model_class = getattr(module, str(model_name))
        model = model_class()
        data_path = model.train_data_path
        data = pd.read_csv(os.path.join(self.base_dir, user, 'Resources', data_path))
        return data
    
    def prepare_base_dir(self):
        if self.base_dir.startswith('~'):
            home = os.path.expanduser('~')
            self.base_dir = os.path.join(home, self.base_dir[2:])
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            

    def get_data(self, version):
        import_path = f"{self.user}.Models.{self.model_name}.{version}"
        module = importlib.import_module(import_path)
        model_class = getattr(module, str(self.model_name))
        model = model_class()
        data_path = model.train_data_path
        data = pd.read_csv(os.path.join(self.base_dir, self.user, 'Resources', data_path))
        return data, model

    def preprocess_data(self, data, ndim):
        y = data[["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"]]
        X = data.drop(columns=["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"])
        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mean(), inplace=True)
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=ndim)
        X = pca.fit_transform(X)
        return X, y

    def run_model(self, ndim, user_triggered, version):
        data, model = self.get_data(version)
        X, y = self.preprocess_data(data, ndim)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model.split_ratio, random_state=42)
        model.run(X_train, y_train, X_test, y_test, ndim)
        result = write_to_json(model, os.path.join(self.base_dir, user_triggered, 'Result'), dimension=ndim, version=version)
        return self.format_result(result)

    def read_params_from_json(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data['params'], data['metrics']
    
    def inference(self, input_data, ndim, user_triggered, version):
        data, model = self.get_data(version)
        X, y = self.preprocess_data(data, ndim)
        input_data = np.array([input_data])
        resource_data = pd.read_csv(os.path.join(self.base_dir, self.user, 'Resources', model.train_data_path))
        resource_X = resource_data.drop(columns=["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"])
        resource_X.fillna(resource_X.mean(), inplace=True)
        # Add the input data to the resource data
        resource_X = resource_X.append(pd.DataFrame(input_data, columns=resource_X.columns))
        resource_X = StandardScaler().fit_transform(resource_X)
        pca = PCA(n_components=ndim)
        resource_X = pca.fit_transform(resource_X)
        
        # Get the last row which is the input data
        input_data = resource_X[-1]
        
        try:
            json_filepath = os.path.join(self.base_dir, user_triggered, 'Result', f'{self.model_name}_pca_{ndim}_{version}.json')
            params, metrics = self.read_params_from_json(json_filepath)
        except:
            raise Exception('Pretrained model has not been found')
        
        # except:
        #     # Train the model if no JSON file is found
        #     data, model = self.get_data()
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model.split_ratio, random_state=42)
        #     model.run(X_train, y_train, X_test, y_test, ndim)
        #     result = write_to_json(model, os.path.join(self.base_dir, user_triggered, 'Result'), dimension=ndim)
        #     params, metrics = self.read_params_from_json(json_filepath)
            
        #     json_filepath = os.path.join(self.base_dir, user_triggered, 'Result', f'{self.model_name}_pca_{ndim}.json')
        #     params, metrics = self.read_params_from_json(json_filepath)
        # Perform inference for each nutrient
        results = {'metrics': metrics}
        for nutrient, param in params.items():
            param_array = np.array(param)
            result = model.inference(input_data, param_array)[0][0]
            nutrient.replace('params_', '')
            results[nutrient] = result.tolist()

        return results

    @staticmethod
    def format_result(result):
        if isinstance(result, np.ndarray):
            return result.tolist()
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
        return result
