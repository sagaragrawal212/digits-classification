import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flask import Flask, request,jsonify
from utils import predict_and_eval
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route("/predict",methods =['POST'])
def predict():
    model_path = "./models/svm_C : 1_gamma : 0.0005.joblib"
    model = load(model_path)
    data = request.get_json()
    data['image1'] = np.array(data['image1']).astype(float)
    data['image2'] = np.array(data['image2']).astype(float)
    
    image_data1 = data['image1'].reshape(1, -1)
    image_data2 = data['image2'].reshape(1, -1)
    print("Read images...")
    # model = load(model_path)
    _,prediction1,_ = predict_and_eval(model , image_data1,np.array([0]) )
    _,prediction2,_ = predict_and_eval(model , image_data2,np.array([0]) )

    print("Prediction 1 : ",prediction1)
    print("Prediction 2 : ",prediction2)
    if prediction1 == prediction2 :
        return "True"
    else :
        return "False" 

if __name__ =="__main__" :
    app.run(host = "0.0.0.0",port = 80)