from flask import Flask
from controller import controller
import configparser
import os 


app = Flask(__name__)

@app.route('/model/<user>/<model_name>/<version>/run', methods=['GET'])
def run_model_route(user, model_name, version):
    return controller.run_model(model_name, user, version)

@app.route('/model/<user>/<model_name>/<version>/inference', methods=['POST'])
def inference_route(user, model_name, version):
    return controller.inference(model_name, user, version)

if __name__ == '__main__':
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
    # app.run(host=host, port=port, debug=True)