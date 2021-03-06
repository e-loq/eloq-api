from flask import Flask, request
from flask_cors import CORS, cross_origin

from base64 import b64encode
from json import dumps

from whole_processing import image_processing

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=["POST"])
def upload_file():
    data = request.get_json()
    file_path = data["file_path"]
    
    image_path, json_path = image_processing(file_path)

    # temp json data
    with open(json_path, "r") as f:
        json_data = f.read()

    # remp image data
    with open(image_path, "rb") as image_file:
        encoded_string = b64encode(image_file.read()).decode('ascii')

    resp_data = {
        "json": json_data,
        "img": [encoded_string, encoded_string]
    }

    # trigger run 
    return dumps(resp_data)

if __name__ == "__main__":
    app.run(debug=True)