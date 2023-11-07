from flask import Flask, request,jsonify
from utils import predict_and_eval
from joblib import load

app = Flask(__name__)

@app.route("/predict",methods =['POST'])
def predict():
    model_path = "./models/svm_C : 0.1_gamma : 0.001.joblib"
    data = request.get_json()
    image_data1 = data['image1']
    image_data2 = data['image2']
    model = load(model_path)
    _,prediction1,_ = predict_and_eval(model , image_data1,None )
    _,prediction2,_ = predict_and_eval(model , image_data2,None )

    if prediction1 == prediction2 :
        return True
    else :
        return False 