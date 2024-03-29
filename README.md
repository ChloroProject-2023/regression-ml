# Flask API for Machine Learning Models

This Flask API provides an interface for running machine learning models and making predictions. It supports dynamic model loading and can process data for different users with customizable PCA dimensionality.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Running the API](#running-the-api)
- [API Usage](#api-usage)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Model Prediction](#model-prediction)
- [Examples](#examples)
  - [Training and Evaluating a Model](#training-and-evaluating-a-model)
  - [Inference Model](#inference-model)
- [Deployment](#deployment)
- [Authors](#authors)


## Getting Started

### Prerequisites

Ensure you have [Python]('https://python.org/download') installed on your machine. The following Python packages are required:

- Flask
- Pandas
- NumPy
- scikit-learn
- Waitress (for Windows deployment)
- Gunicorn (for Linux deployment)

### Installation

To set up the project, start by cloning the repository or downloading the source code. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Configuration
The API server's host and port are configurable via a config.ini file. Create this file in your project directory and populate it. Default values are:
```ini
[DEFAULT]
PORT = 5000
HOST = 0.0.0.0
BASE_DIR = ~/GroupProject
```

## Running the API
To run the API, navigate to the project directory and execute the api.py script:
```bash
python api.py
```
The API will be available at http://localhost:5000 by default.

## API Usage
The API provides several endpoints:
## Model Training and Evaluation

- **Endpoint**: `/model/<user>/<model_name>/<version>/run`
- **Method**: `GET`
- **Description**: Trains and evaluates the specified model for a given user.
- **Query Parameters**:
  - `ndim` (optional): The number of dimensions for dimension reduction.
  - `user` (optional): Identifier for the user who will use the model. If not provided, the user who created the model will be used.
- **path Parameters**:
  - `user`: Identifier for the user who own the model.
  - `model_name`: Name of the model to be trained and evaluated.
  - `version`: Version of model

## Model Prediction

- **Endpoint**: `/model/<user>/<model_name>/<version>/inference`
- **Method**: `POST`
- **Description**: Uses the specified model to make 
predictions.
- **Body Parameters**:
  - `input`: A list (array) of number which has the same element length match the train dataset.
  - `user_id` (optional): Identifier for the user who will use the model. If not provided, the user who created the model will be used.
- **Path parameters**:
  - `user`: Identifier for the user who own the model.
  - `model_name`: Name of the model to be trained and evaluated.
  - `version`: Version of model
- **Query parameters**:
  - `ndim` (optional): The number of dimensions for dimension reduction.

## Examples

### Training and Evaluating a Model


To train and evaluate a model for a user id 'MPAP' using the Linear Regression model with 3 PCA dimensions, send a GET request to the following endpoint:

**Request:**

```http
GET /model/MPAP/Linear_Regression/run?ndim=3&user=MPAP
```

**Response:**

```json
{
    "message": "success",
    "result" : "path/to/base/MPAP/Results/Linear_Regression_pca_3_v1.json"
}
```

### Inference model
To make predictions using the Linear Regression model for a user id 'MPAP' with 3 PCA dimensions. The model must be trained and evaluated first.

**Request:**

```http
POST /model/MPAP/Linear_Regression/inference?ndim=3&user=MPAP
Content-Type: application/json

{
    "input": [1, 0.5, 2.5, 1.2 ... 0.5]
}
```

**Response:**

```json
{
    "message": "success",
    "result": {
        "K": 40387.50892440533,
        "N": 3042.7882326102417,
        "P": 4409.603387604054,
        "metrics": {
            "mse": 65467184.53838297,
            "r2": 0.00751873517956847
        }
    }
}
```

## Deployment
The API can be deployed on a Windows machine using Waitress. To do so, install Waitress and run the following command:

**For Windows**
```bash
python api.py
```
**For Linux**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```
## Authors
- [**BUI, Huy Hoang**](https://github.com/bhhoang)
- [**DAO, Duy Manh Ha**](https://github.com/R1verrrr)
