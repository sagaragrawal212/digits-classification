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
    return 'Digit Classification Model Deployment'

# @app.route("/predict",methods =['POST'])
# def predict():
#     model_path = "./models/svm_C : 0.1_gamma : 0.0001.joblib"
#     model = load(model_path)
#     data = request.get_json()
#     data['image'] = np.array(data['image']).astype(float)
    
#     image_data = data['image'].reshape(1, -1)
    
#     _,prediction,_ = predict_and_eval(model , image_data,np.array([0]))
#     return ({"Prediction": str(prediction[0])})

# Global variables
svm_model = None
lr_model = None
tree_model = None

def load_model():
    global svm_model, lr_model, tree_model

    # Load SVM model
    svm_model = load("./models/svm_C : 0.1_gamma : 0.0001.joblib")

    # Load Logistic Regression model
    lr_model = load("./models/M22AIE212_lr_lbfgs.joblib")

    # Load Decision Tree model
    tree_model = load("./models/decision_tree_criterion : entropy_max_depth : 20.joblib")

# Call load_model() when the application starts
load_model()

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    
    if model_type == "svm" : 
        model = svm_model
    elif model_type == "decision_tree":
        model = tree_model
    elif model_type == "lr" :
        model = lr_model

    data = request.get_json()
    data['image'] = np.array(data['image']).astype(float)
    
    image_data = data['image'].reshape(1, -1)
    
    _,prediction,_ = predict_and_eval(model , image_data,np.array([0]))
    return ({f"{model_type.upper()} Prediction": str(prediction[0])})

# if __name__ =="__main__" :
#     app.run(host = "0.0.0.0",port = 80)
    
