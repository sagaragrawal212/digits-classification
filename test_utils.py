from utils import *

def test_for_hparam_combinations_count():

    gamma_list = [0.001,0.01,0.1,1,10,100]
    C_ranges = [0.1 ,1,2,5,10]  
    h_params_combinations = get_hyperparameter_combinations(gamma_list,C_ranges)

    assert len(h_params_combinations) == len(gamma_list)*len(C_ranges)


def test_for_hparam_combinations_value():

    gamma_list = [0.001,0.01]
    C_ranges = [0.1]  
    h_params_combinations = get_hyperparameter_combinations(gamma_list,C_ranges)

    expected_param_combo_1 = {}
    expected_param_combo_2 = {}
    assert  (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

def test_data_splitting():

    X,y = read_digits()
    X =     X[:100,:,:]
    y = y[:100]

    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - (test_size + dev_size)

    X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = test_size,dev_size = dev_size)

    assert ((len(X_train) == int(train_size*len(X))) &
            (len(X_dev) == int(dev_size*len(X))) &
            (len(X_test) == int(test_size*len(X))))
    