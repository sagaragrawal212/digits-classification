"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt
import itertools
# Import datasets, classifiers and performance metrics
import pdb
from utils import *


# Get the dataset :
X , y = read_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


print("Iterating for different test and dev size : ")
dev_splits = [0.1,0.2,0.3]
test_splits =[0.1,0.2,0.3]
test_dev_list = [dev_splits,test_splits]
all_test_dev_combination = all_param_comb_list = list(itertools.product(*test_dev_list)) 

for dev_size , test_size in all_test_dev_combination :
    # Data Splitting -- to create train and test sets
    X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = test_size,dev_size = dev_size)
   
    # Data prepoocessing
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)


    #HYPER PARAMETER TUNING 
    gamma_ranges = [0.001,0.01,0.1,1,10,100]
    C_ranges = [0.1 ,1,2,5,10]

    
    list_of_all_param_combination = get_hyperparameter_combinations(gamma_ranges,C_ranges)

    optimal_params , best_model , best_acc_so_far = tune_hparams(X_train,y_train,X_dev,y_dev,list_of_all_param_combination) 

    # print(f"Optimal para gamma = {optimal_params} ")
    # Model training : Create a classifier: a support vector classifier
    # model = train_model(X_train, y_train , {'gamma' : optimal_gamma , 'C' : optimal_C} , model_type = 'svm')

    # 6. Getting the predictions on test set
    train_acc = predict_and_eval(best_model , X_train , y_train ) 
    dev_acc = predict_and_eval(best_model , X_dev , y_dev ) 
    test_acc = predict_and_eval(best_model , X_test , y_test )
    print() 
    print(f"test_size = {test_size} ,  dev_size = {dev_size} ,  train_size  = {1 - (test_size + dev_size)} train_acc = {train_acc} dev_acc = {dev_acc}  test_acc == {test_acc} ")
    print(f"Optimal Parameters : {optimal_params}")