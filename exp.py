"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import pdb
from utils import *

gamma_ranges = [0.001,0.01,0.1,1,10,100]
C_ranges = [0.1 , 1 , 2 , 5 , 10 ]

# Get the dataset :
X , y = read_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Data Splitting -- to create train and test sets
X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = 0.3,dev_size = 0.2)
print(f"Train data size : {len(X_train)}")
print(f"Dev data size : {len(X_test)}")
print(f"Val data size : {len(X_dev)}")

# Data prepoocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

#HYPER PARAMETER TUNING 
best_acc_so_far = -1
best_model = None
for cur_gamma in gamma_ranges :
    for cur_C in C_ranges :
       
        cur_model  = train_model(X_train,y_train, {'gamma' : cur_gamma , 'C' : cur_C}, model_type = 'svm')
        cur_accuracy = predict_and_eval(cur_model,X_dev,y_dev)

        if cur_accuracy > best_acc_so_far :
            print("New best accuracy : ",cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_C = cur_C
            best_model = cur_model

print(f"Optimal para gamma = {optimal_gamma} and C = {optimal_C}")
# Model training : Create a classifier: a support vector classifier
# model = train_model(X_train, y_train , {'gamma' : optimal_gamma , 'C' : optimal_C} , model_type = 'svm')

# 6. Getting the predictions on test set
test_acc = predict_and_eval(best_model , X_test , y_test ) 
print("Test Accuracy : ",test_acc)