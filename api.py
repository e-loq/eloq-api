from flask import Flask, request, jsonify
import sys
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=["POST"])
def upload_file():
    print("asdf")
    data = request.get_json()
    url = data["url"]
    return jsonify(data) 

if __name__ == "__main__":
    app.run(debug=True)