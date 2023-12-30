from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import importlib
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.write_json import write_to_json
import configparser

app = Flask(__name__)

@app.route('/', methods=['GET'])
def instructions():
    return (
        f"Welcome to the API! Here's how to use it:\n"
    )

@app.route('/model/<user>/<model_name>/run', methods=['GET'])
def run_model(model_name, user):
    # Construct the import path
    ndim = 3
    query = request.args
    user_triggered = user
    if query:
        ndim = int(query['ndim'])
        user_triggered = str(query['user']) if 'user' in query else user
                
    import_path = f"{user}.{model_name}"

    # Dynamically import the model
    module = importlib.import_module(import_path)

    # Assuming the class name in LinearRegression.py is LinearRegressionModel
    # Replace 'LinearRegressionModel' with the actual class name if it's different
    model_class = getattr(module, str(model_name))
    model = model_class()

    # Assuming the model has an attribute 'train_data_path' specifying the data file name
    data_path = model.train_data_path

    # Read the CSV file from the user's Resources directory
    data = pd.read_csv(f'./{user}/Resources/{data_path}')

    # Assuming 'P conc. (mg/kg)', 'N conc. (mg/kg)', and 'K conc. (mg/kg)' are the target columns
    # And the rest are feature columns
    y = data[["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"]]
    X = data.drop(columns=["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"])
    
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    X  = StandardScaler().fit_transform(X)
    pca = PCA(n_components=ndim)
    X  = pca.fit_transform(X)
    if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
           os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model.split_ratio, random_state=42)
    # Run the model (assuming the model has a method named 'run')
    model.run(X_train, y_train, X_test, y_test, ndim)
    
    result = write_to_json(model, os.path.join(os.getcwd(), user_triggered, 'Result'), dimension=ndim)
    
   # Convert Numpy arrays in 'result' to lists before returning
    if isinstance(result, np.ndarray):
        result = result.tolist()  # Convert entire array to list if result is a Numpy array
        
    # If result is a dictionary containing Numpy arrays, convert them individually
    elif isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
    if not result:
        return jsonify({'message': 'error', 'error': 'Result not found'}), 512, {'ContentType':'application/json'}

    return jsonify({'message': 'success', 'result': result}), 200, {'ContentType':'application/json'}

@app.route('/model/<user>/<model_name>/inference', methods=['POST'])
def inference(model_name, user):
        # Construct the import path
    ndim = 3
    query = request.args
    user_triggered = user   
    if query:
        ndim = int(query['ndim'])
        
    body = request.get_json()
    input = body['input']
    user_triggered = str(body['user_id']) if 'user_id' in body else user
    import_path = f"{user}.{model_name}"
    # Check body
    if not input:
        raise ValueError('Missing input')
    if not isinstance(input, list):
        raise ValueError('Input must be a list')
    # Dynamically import the model
    module = importlib.import_module(import_path)
    # PCA input
    try:
        input = np.array([input])
        input = StandardScaler().fit_transform(input)
        pca = PCA(n_components=ndim)
        input = pca.fit_transform(input)
    except Exception as e:
        # Status code 400  for bad request
        return jsonify({'message': 'error', 'error': str(e)}) , 400, {'ContentType':'application/json'}
    # Assuming the class name in LinearRegression.py is LinearRegressionModel
    # Replace 'LinearRegressionModel' with the actual class name if it's different
    model_class = getattr(module, str(model_name))
    model = model_class(user)

    # Assuming the model has an attribute 'train_data_path' specifying the data file name
    data_path = model.train_data_path

    # Read the CSV file from the user's Resources directory
    data = pd.read_csv(f'./{user}/Resources/{data_path}')

    # Assuming 'P conc. (mg/kg)', 'N conc. (mg/kg)', and 'K conc. (mg/kg)' are the target columns
    # And the rest are feature columns
    y = data[["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"]]
    X = data.drop(columns=["P conc. (mg/kg)", "N conc. (mg/kg)", "K conc. (mg/kg)"])
    
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    X  = StandardScaler().fit_transform(X)
    pca = PCA(n_components=ndim)
    X  = pca.fit_transform(X)
    if not os.path.exists(os.path.join(os.getcwd(), user, 'Result')):
           os.makedirs(os.path.join(os.getcwd(), user, 'Result'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model.split_ratio, random_state=42)
    # Run the model (assuming the model has a method named 'run')
    model.run(X_train, y_train, X_test, y_test, ndim, os.path.join(os.getcwd(), user_triggered, 'Result'))
    
    result = model.predictNPK(input)
    
    # If no result, return error message
    if result is None:
        return jsonify({'message': 'error', 'error': 'Result not found'}), 512, {'ContentType':'application/json'}
    
    return jsonify({'message': 'success', 'result': result}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':
    #Get host and 
    config = configparser.ConfigParser()
    config.read('config.ini')
    host = config['DEFAULT']['HOST']
    port = config['DEFAULT']['PORT']
    if not os.name == 'nt':
        print(f"Run 'gunicorn -w 4 -b {host}:{port} api:app' to start the server on Unix/Linux.")
        exit()
    from waitress import serve
    print(f'Running on {host}:{port}')
    serve(app, host=host, port=port)