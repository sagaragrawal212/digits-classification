from utils import *
import os
from sklearn.model_selection import ParameterGrid

def test_for_hparam_combinations_count():

  
    gamma_list , C_ranges,h_params_combinations =dummy_hyper_parameters()

    assert len(h_params_combinations) == len(gamma_list)*len(C_ranges)

def dummy_hyper_parameters():
    
    gamma_list = [0.001,0.01,0.1,1,10,100]
    C_ranges = [0.1 ,1,2,5,10]  
    param_grid = {'gamma': [0.001,0.01,0.1,1,10,100], 'C': [0.1 ,1,2,5,10]}
    h_params_combinations =ParameterGrid(param_grid)

    return gamma_list , C_ranges,h_params_combinations

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