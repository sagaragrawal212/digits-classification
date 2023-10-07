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
from joblib import load
from sklearn.model_selection import ParameterGrid
import sys

# python exp.py -max_run,-dev_size,

# max_run =sys.argv[1] 
# Get the dataset :

import argparse

parser = argparse.ArgumentParser(description = """
                      max_run : maximum number of model runs
                      test_size : test Dataset size
                      dev_size : dev Dataset Size
                      model_type : name of model
                      """)
parser.add_argument('-runs',
            '--max_run',
            type = int,
            default = 5,
            help = 'maximum number of model runs')

parser.add_argument('-test_size',
            '--test_size',
            type = float,
            default = 0.1,
            help = 'test Dataset size')
parser.add_argument('-dev_size',
            '--dev_size',
            type = float,
            default = 0.1,
            help = 'dev Dataset Size')
parser.add_argument('-model',
            '--model_type',
            type = str,
            default = 'svm',
            help = 'name of model')
args = parser.parse_args()
X , y = read_digits()

print("Total Number of samples in dataset : ",len(X))
print("Size of image : ",X[0].shape)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Model 1 : SVM
param_grid = {'gamma': [0.001,0.01,0.1,1,10,100], 'C': [0.1 ,1,2,5,10]}
list_of_all_param_combination_svm = ParameterGrid(param_grid)

# Model 2 : Decision Tree
param_grid = {'criterion':['gini','entropy'],
            'max_depth':[10,20,30,40,50,60,70,80,90,100]
            }
list_of_all_param_combination_dt = ParameterGrid(param_grid)

hparam_dict = {'svm' : list_of_all_param_combination_svm ,
                    'decision_tree' : list_of_all_param_combination_dt }


print("Iterating for different test and dev size : ")
dev_splits = [args.test_size]
test_splits =[args.dev_size]
test_dev_list = [dev_splits,test_splits]
all_test_dev_combination = all_param_comb_list = list(itertools.product(*test_dev_list)) 
max_run = args.max_run

for i in range(max_run) :
    print()
    print(f"Iteration Number : {i}")
    for model_type in [args.model_type] :
        for dev_size , test_size in all_test_dev_combination :
            # Data Splitting -- to create train and test sets
            X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = test_size,dev_size = dev_size)
        
            # Data prepoocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            optimal_params ,best_model_path, best_acc_so_far = tune_hparams(X_train,y_train,X_dev,y_dev,hparam_dict[model_type],model_type = model_type) 
            
            best_model = load(best_model_path)
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