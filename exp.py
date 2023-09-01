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

# Get the dataset :
X , y = read_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Data Splitting -- to create train and test sets
X_train, X_test,X_dev, y_train, y_test,y_dev = split_data(X , y,test_size = 0.2,dev_size = 0.25)
print(f"Train data size : {len(X_train)}")
print(f"Dev data size : {len(X_test)}")
print(f"Val data size : {len(X_dev)}")

# Data prepoocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
    
# Model training : Create a classifier: a support vector classifier
model = train_model(X_train, y_train , {'gamma' : 0.001} , model_type = 'svm')

# 6. Getting the predictions on test set
y_pred = predict_and_eval(model , X_test , y_test ) 