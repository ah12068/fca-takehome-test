import flask
import json
import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
 response = json.dumps({'response': 'yahhhh!'})
 return response, 200
if __name__ == '__main__':
 application.run(debug=True)
