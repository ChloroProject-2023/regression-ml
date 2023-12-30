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

- **Endpoint**: `/model/<user>/<model_name>/run`
- **Method**: `GET`
- **Description**: Trains and evaluates the specified model for a given user.
- **Query Parameters**:
  - `ndim` (optional): The number of dimensions for PCA.
  - `user` (optional): Identifier for the user.

## Model Prediction

- **Endpoint**: `/model/<user>/<model_name>/interference`
- **Method**: `POST`
- **Description**: Uses the specified model to make 
predictions.
- **Body Parameters**:
  - `input`: A list of input features.
  - `user_id` (optional): Identifier for the user. If not provided, the user who created the model will be used.

## Examples

### Training and Evaluating a Model

To train and evaluate a model for a user named 'john_doe' using the Linear Regression model with 3 PCA dimensions, send a GET request to the following endpoint:

**Request:**

```http
GET /model/MPAP/Linear_Regression/run?ndim=3
```

**Response:**

```json
{
    "message": "success",
    "result" : "path/to/result"
}
```

### Inference model
To make predictions using the Linear Regression model for a user named 'MPAP':

**Request:**

```http
POST /model/MPAP/Linear_Regression/interference
Content-Type: application/json

{
    "input": [1, 0.5, 2.5, 1.2 ... 0.5],
    "user_id": "MPAP"
}
```
**Response:**

```json
{
    "message": "success",
    "result" : {
       "N": 1,
       "P": 1,
       "K": 1
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