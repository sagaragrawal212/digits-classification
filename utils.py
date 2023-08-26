from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# will keep files in utils.py
def preprocess_data(digit_image) :

    # pdb.set_trace()
    # flatten the images
    n_samples = len(digit_image)
    data = digit_image.reshape((n_samples, -1))

    return data 

# perform train - test split of 50 - 50 %
def split_data(X, y , test_size , random_state = 1) :
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=test_size,random_state = random_state)
    return X_train, X_test, y_train, y_test

# Train the model of choice with the model_type parameters
def train_model(X , y , model_params , model_type = "svm") :

    if model_type == "svm" :
        clf = svm.SVC
    
    model = clf(**model_params)
    
    model.fit(X, y)

    return model

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target

    return X , y