from flask import request, jsonify
from services.reflectence import ReflectenceService 

def run_model(model_name: str, user: str, version: str) -> list:
    """Train the model

    Args:
        model_name (str): Name of the model
        user (str): User who create the model
        version (str): Version of the model

    Returns:
        list: List of objects for flask
    """
    # Extract query parameters if any
    query = request.args
    ndim = int(query['ndim']) if 'ndim' in query else 3
    user_triggered = str(query['user']) if 'user' in query else user

    # Create an instance of ModelService
    service = ReflectenceService(user, model_name, ndim, version)

    try:
        # Run the model using the service
        result = service.run_model(ndim, user_triggered, version)
        return jsonify({'message': 'success', 'result': result}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'message': 'error', 'error': str(e)}), 400, {'Content-Type': 'application/json'}


def inference(model_name, user, version):
    """Inference pretrained model

    Args:
        model_name (str): Name of the model
        user (str): User who create the model
        version (str): Version of the model

    Returns:
        list: List of objects for flask
    """
    # Extract and validate JSON body from the request
    body = request.get_json()
    query = request.args
    if not body or 'input' not in body:
        return jsonify({'message': 'error', 'error': 'Missing input'}), 400, {'Content-Type': 'application/json'}

    input_data = body['input']
    ndim = int(query['ndim']) if 'ndim' in query else 1
    user_triggered = str(body['user_id']) if 'user_id' in body else user

    # Create an instance of ModelService
    service = ReflectenceService(user, model_name, ndim, version)

    try:
        # Run inference using the service
        result = service.inference(input_data, ndim, user_triggered, version)
        return jsonify({'message': 'success', 'result': result}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'message': 'error', 'error': str(e)}), 400, {'Content-Type': 'application/json'}
