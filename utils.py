from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from joblib import dump


def preprocess_data(digit_image) :

    '''
    flatten the images

    Returns : Flattened data
    '''

    n_samples = len(digit_image)
    data = digit_image.reshape((n_samples, -1))

    return data 

def split_data(X, y , test_size ,dev_size , random_state = 1) :
    
    """
    First split the data into train_dev and test sets

    Next split the train_dev set to train and dev sets
    """
    
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X,y, test_size=test_size,random_state = random_state)
 
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev,y_train_dev, test_size=dev_size ,random_state = random_state)

    
    return X_train, X_test,X_dev, y_train, y_test,y_dev



def train_model(X , y, model_params ,model_type = "svm" ) :
    '''
    Train the model of choice with the model_type parameters

            Parameters:
                   X : train data matrix
                   y : train data ground truths
                   model_params : model parameters
                   model_type : type of model to be trained

    Returns : model: trained classification model
    '''

    if model_type == "svm" :
        clf = svm.SVC
    elif model_type == "decision_tree":
        clf = tree.DecisionTreeClassifier

    model = clf(**model_params)
    
    model.fit(X, y)

    return model

def read_digits():
    '''
    Loads the digits dataset using sklearn datasets

    Returns : X (features) and y (labels) corresponding to MNIST dataset
    '''
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    return X , y

def predict_and_eval(model , X_test , y_test ) :
    
    '''
    Predict the value of the digit on the test subset and evaluates the model performance.

            Parameters:
                   model : trained classification model
                   X_test : test data matrix
                   y_test : test data ground truths
                

            Returns:
                    predictions of the model and prints the evaluation metrics
    '''
    # 
    predicted = model.predict(X_test)

    # ###############################################################################
    # # Below we visualize the first 4 test samples and show their predicted
    # # digit value in the title.

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")

    # ###############################################################################
    # # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # # the main classification metrics.

    # print(
    #     f"Classification report for classifier {model}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )


    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()

    # # The ground truth and predicted lists
    # y_true = []
    # y_pred = []
    # cm = disp.confusion_matrix

    # # For each cell in the confusion matrix, add the corresponding ground truths
    # # and predictions to the lists
    # for gt in range(len(cm)):
    #     for pred in range(len(cm)):
    #         y_true += [gt] * cm[gt][pred]
    #         y_pred += [pred] * cm[gt][pred]

    # print(
    #     "Classification report rebuilt from confusion matrix:\n"
    #     f"{metrics.classification_report(y_true, y_pred)}\n"
    # )

    test_acc = metrics.accuracy_score(y_test,predicted)
    
    # print("Test Accuracy : ",test_acc)
    return test_acc , predicted ,y_test


def tune_hparams(X_train,y_train,X_dev,y_dev,list_of_all_param_combination,model_type = 'svm') :
    
    best_acc_so_far = -1
    best_model = None

    for param_combination in list_of_all_param_combination :
        
        cur_model  = train_model(X_train,y_train,  param_combination,model_type = model_type)
        cur_accuracy,_,_ = predict_and_eval(cur_model,X_dev,y_dev)

        if cur_accuracy > best_acc_so_far :
            print("New best accuracy : ",cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_params = param_combination
            best_model = cur_model
            best_model_path = model_type + "_" + "_".join([f'{k} : {v}' for k,v in optimal_params.items()]) + ".joblib"
        dump(best_model,best_model_path)
    return optimal_params , best_model_path , best_acc_so_far

