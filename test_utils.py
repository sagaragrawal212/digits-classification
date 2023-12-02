from utils import *
import os
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.linear_model import LogisticRegression
np.random.seed(42)
# Creating models directory
if os.path.exists("models"):
    pass
else :
    os.system("mkdir models")


def test_for_hparam_combinations_count():

  
    gamma_list , C_ranges,h_params_combinations =dummy_hyper_parameters()

    assert len(h_params_combinations) == len(gamma_list)*len(C_ranges)

def dummy_hyper_parameters():
    
    gamma_list = [0.001,0.01,0.1,1,10,100]
    C_ranges = [0.1 ,1,2,5,10]  
    param_grid = {'gamma': [0.001,0.01,0.1,1,10,100], 'C': [0.1 ,1,2,5,10]}
    h_params_combinations =ParameterGrid(param_grid)

    return gamma_list , C_ranges,h_params_combinations

def dummy_hyper_lr_parameters():
    
    solver = ["lbfgs", "liblinear","newton-cg", "newton-cholesky", "sag", "saga"]
    param_grid = {'solver': solver}
    h_params_combinations =ParameterGrid(param_grid)

    return h_params_combinations

def create_dummy_data() :
    
    X , y = read_digits()

    X_train = X[:100,:,:]
    X_dev = X[:50,:,:]
    y_train = y[:100]
    y_dev = y[:50]
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)

    return X_train,y_train,X_dev,y_dev
# def test_for_hparam_combinations_value():

#     gamma_list = [0.001,0.01]
#     C_ranges = [0.1]  
#     h_params_combinations = get_hyperparameter_combinations(gamma_list,C_ranges)

#     expected_param_combo_1 = {}
#     expected_param_combo_2 = {}
#     assert  (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

# def test_data_splitting():

#     X,y = read_digits()
#     X =     X[:100,:,:]
#     y = y[:100]

#     test_size = 0.1
#     dev_size = 0.6
#     train_size = 1 - (test_size + dev_size)

#     X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = test_size,dev_size = dev_size)

#     assert ((len(X_train) == int(train_size*len(X))) &
#             (len(X_dev) == int(dev_size*len(X))) &
#             (len(X_test) == int(test_size*len(X))))

def test_model_saving() :
    X_train,y_train,X_dev,y_dev = create_dummy_data()
    _,_,list_of_all_param_combination = dummy_hyper_parameters() 
    _ ,best_model_path , _ = tune_hparams(X_train,y_train,X_dev,y_dev,list_of_all_param_combination)

    assert os.path.exists(best_model_path)

from api.app import app
def test_get_root() :
    response = app.test_client().get("/")
    assert response.status_code == 200 
    assert response.get_data() == b'Digit Classification Model Deployment'


# def test_post_predict():
    
#     X,y = read_digits()
#     X_reshape = preprocess_data(X)

#     ## Digit 0 test
#     indices_digit_0 = np.where(y == 0)[0]
#     index_digit_0 = np.random.choice(indices_digit_0)
#     image_data_0 = list(X_reshape[index_digit_0].astype(str))
    
#     response = app.test_client().post("/predict", json={"image":image_data_0})

#     ## Status Code Check
#     assert response.status_code == 200 
#     assert response.get_json()['Prediction'] == '0'

#     ## Digit 1 test
#     indices_digit_1 = np.where(y == 1)[0]
#     index_digit_1 = np.random.choice(indices_digit_1)
#     image_data_1 = list(X_reshape[index_digit_1].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_1})
#     assert response.get_json()['Prediction'] == '1'

#     ## Digit 2 test
#     indices_digit_2 = np.where(y == 2)[0]
#     index_digit_2 = np.random.choice(indices_digit_2)
#     image_data_2 = list(X_reshape[index_digit_2].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_2})
#     assert response.get_json()['Prediction'] == '2'

#     ## Digit 3 test
#     indices_digit_3 = np.where(y == 3)[0]
#     index_digit_3 = np.random.choice(indices_digit_3)
#     image_data_3 = list(X_reshape[index_digit_3].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_3})
#     assert response.get_json()['Prediction'] == '3'

#     ## Digit 4 test
#     indices_digit_4 = np.where(y == 4)[0]
#     index_digit_4 = np.random.choice(indices_digit_4)
#     image_data_4 = list(X_reshape[index_digit_4].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_4})
#     assert response.get_json()['Prediction'] == '4'
   
#     ## Digit 5 test
#     indices_digit_5 = np.where(y == 5)[0]
#     index_digit_5 = np.random.choice(indices_digit_5)
#     image_data_5 = list(X_reshape[index_digit_5].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_5})
#     assert response.get_json()['Prediction'] == '5'

#     ## Digit 6 test
#     indices_digit_6 = np.where(y == 6)[0]
#     index_digit_6 = np.random.choice(indices_digit_6)
#     image_data_6 = list(X_reshape[index_digit_6].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_6})
#     assert response.get_json()['Prediction'] == '6'

#     ## Digit 7 test
#     indices_digit_7 = np.where(y == 7)[0]
#     index_digit_7 = np.random.choice(indices_digit_7)
#     image_data_7 = list(X_reshape[index_digit_7].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_7})
#     assert response.get_json()['Prediction'] == '7'

#     ## Digit 8 test
#     indices_digit_8 = np.where(y == 8)[0]
#     index_digit_8 = np.random.choice(indices_digit_8)
#     image_data_8 = list(X_reshape[index_digit_8].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_8})
#     assert response.get_json()['Prediction'] == '8'

#     ## Digit 9 test
#     indices_digit_9 = np.where(y == 9)[0]
#     index_digit_9 = np.random.choice(indices_digit_9)
#     image_data_9 = list(X_reshape[index_digit_9].astype(str))
#     response = app.test_client().post("/predict", json={"image":image_data_9})
#     assert response.get_json()['Prediction'] == '9'

def test_model_lr_loading() :
    X_train,y_train,X_dev,y_dev = create_dummy_data()
    list_of_all_param_combination = dummy_hyper_lr_parameters() 
    _ ,best_model_path , _ = tune_hparams(X_train,y_train,X_dev,y_dev,list_of_all_param_combination,model_type = 'lr')

    loaded_model = load(best_model_path)
    assert isinstance(loaded_model, LogisticRegression), f"Model loaded from {best_model_path} is not a Logistic Regression model"



def test_model_lr_loading_solver() :
    X_train,y_train,X_dev,y_dev = create_dummy_data()
    list_of_all_param_combination = dummy_hyper_lr_parameters() 
    _ ,best_model_path , _ = tune_hparams(X_train,y_train,X_dev,y_dev,list_of_all_param_combination,model_type = 'lr')

    loaded_model = load(best_model_path)
    
    assert loaded_model.solver == best_model_path.split("_")[-1].split(".")[0].split(":")[-1].strip(), f"Solver name in the model file name does not match the solver used in the loaded model ({loaded_model.solver})"