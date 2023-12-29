from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import importlib
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)

@app.route('/model/<user>/<model_name>/run', methods=['GET'])
def run_model(model_name, user):
    # Construct the import path
    ndim = 3
    query = request.args
    if query:
        ndim = int(query['ndim'])
                
    import_path = f"{user}.{model_name}"

    # Dynamically import the model
    module = importlib.import_module(import_path)

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model.split_ratio, random_state=42)
    # Run the model (assuming the model has a method named 'run')
    result = model.run(X_train, y_train, X_test, y_test, ndim)
    
   # Convert Numpy arrays in 'result' to lists before returning
    if isinstance(result, np.ndarray):
        result = result.tolist()  # Convert entire array to list if result is a Numpy array
        
    # If result is a dictionary containing Numpy arrays, convert them individually
    elif isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()

    return jsonify({'message': 'success', 'result': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
