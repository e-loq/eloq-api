from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=["POST"])
def upload_file():
    data = request.get_json()
    file_path = data["file_path"]
    print(file_path)
    
    # trigger run 
    return ""

if __name__ == "__main__":
    app.run(debug=True)