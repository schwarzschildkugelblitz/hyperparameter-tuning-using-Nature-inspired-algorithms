import numpy
import math
import random
import tensorflow as tf 
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

# def ELM_tune():
#     param_ranges = {
#     'hidden_size': [32, 128],
#     'activation': ['sigmoid', 'relu']
#     }

#     hidden_size, activation = parameters
    
#     # Convert the hyperparameter strings to the appropriate data types
#     activation = eval(activation)
    
#     # Create and train the ELM model
#     model = ELM(hidden_size=hidden_size, activation=activation)
#     model.fit(X_train, y_train)
    
#     # Return the model's validation accuracy
#     return model.score(X_val, y_val)

# def SVM():

# def FNN():

def LSTM_(search_agent,X_train,X_val,y_train, y_val,Task,num_class):
    # param_ranges = {
    # 'num_epochs': [10, 100],
    # 'batch_size': [32, 256],
    # 'learning_rate': [1e-3, 1e-1],
    # 'hidden_size': [16, 256],
    # 'num_layers': [1, 6],
    # 'dropout': [0, 0.]
    # }
    print(X_train.shape)
    if Task == "Regression":
        Loss = 'RSME'
        metric =["r_square","MAE"]
    elif Task == "Binary_classification" :
        Loss = 'binary_crossentropy'
        metric =["accuracy",tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),tf.keras.metrics.F1Score()]
    else :
        Loss = 'categorical_crossentropy'
        metric =["accuracy",tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]


    num_epochs = int(search_agent[0])
    batch_size = int(search_agent[1])
    learning_rate = float(search_agent[2])
    hidden_size = int(search_agent[3])
    num_layers = int(search_agent[4])
    dropout = float(search_agent[5])

    model = Sequential()
    for i in range (num_layers-1):
        model.add(LSTM(hidden_size, dropout=dropout,return_sequences=True))
    model.add(LSTM(hidden_size, dropout=dropout,return_sequences=False))
    if Task != "Regression":
        model.add(Dense(num_class, activation = "softmax"))
    else :
        model.add(Dense(1))
    model.compile(loss=Loss, optimizer=Adam(learning_rate), metrics=metric)
    y_train= to_categorical(y_train-1,num_class)
    y_val= to_categorical(y_val-1,num_class)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)
    a = model.evaluate(X_val, y_val)
    if a[2]==0  or a[3] == 0 :
        a.append(0)
    else : 
        a.append((2*a[2]*a[3])/(a[2]+a[3]))
    print(a)
    numpy.array(a)
    return a

# def GRNN_tune():
#     param_ranges = {
#     'sigma': [0.1, 1.0]
#     }
#     sigma = parameters
    
#     # Create and train the GRNN model
#     model = GRNN(sigma=sigma)
#     model.fit(X_train, y_train)
    
#     # Return the model's validation accuracy
#     return model.score(X_val, y_val)


# def SVM_tune():
#     param_ranges = {
#     'C': [0.1, 10.0],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
#     }
#     C, kernel, gamma = parameters
    
#     # Convert the hyperparameter strings to the appropriate data types
#     kernel = eval(kernel)
#     gamma = eval(gamma)
    
#     # Create and train the SVM model
#     model = SVC(C=C, kernel=kernel, gamma=gamma)
#     model.fit(X_train, y_train)
    
#     # Return the model's validation accuracy
#     return model.score(X_val, y_val)


def getFunctionDetails(a):
    param = {
        "LSTM_": ["LSTM_",[10,32,1e-3,16,1,0],[100,256,1e-1,256,1,0.9],6],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],

    }
    return param.get(a, "nothing")