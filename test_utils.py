from utils import get_hyperparameter_combinations

def test_for_hparam_combinations_count():

    gamma_list = [0.001,0.01,0.1,1,10,100]
    C_ranges = [0.1 ,1,2,5,10]  
    h_params_combinations = get_hyperparameter_combinations(gamma_list,C_ranges)

    assert len(h_params_combinations) == len(gamma_list)*len(C_ranges)


# def test_for_hparam_combinations_value():

#     assert  (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

