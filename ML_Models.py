import numpy
import math
import random
import tensorflow as tf


def ELM_tune():
    param_ranges = {
    'hidden_size': [32, 128],
    'activation': ['sigmoid', 'relu']
    }

    hidden_size, activation = parameters
    
    # Convert the hyperparameter strings to the appropriate data types
    activation = eval(activation)
    
    # Create and train the ELM model
    model = ELM(hidden_size=hidden_size, activation=activation)
    model.fit(X_train, y_train)
    
    # Return the model's validation accuracy
    return model.score(X_val, y_val)

def SVM():

def FNN():

def LSTM_tune():
    param_ranges = {
    'num_epochs': [10, 50],
    'batch_size': [32, 256],
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [1e-3, 1e-1],
    'hidden_size': [32, 128],
    'num_layers': [1, 3],
    'dropout': [0, 0.5]
}
    num_epochs, batch_size, optimizer, learning_rate, hidden_size, num_layers, dropout = parameters
    
    # Convert the hyperparameter strings to the appropriate data types
    optimizer = eval(optimizer)
    learning_rate = float(learning_rate)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    dropout = float(dropout)
    
    # Create and train the LSTM model
    model = LSTM(hidden_size, num_layers, dropout=dropout)
    model.compile(loss='binary_crossentropy', optimizer=optimizer(learning_rate), metrics=['RSME'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))
    return model.evaluate(X_val, y_val)[1],

def GRNN_tune():
    param_ranges = {
    'sigma': [0.1, 1.0]
    }
    sigma = parameters
    
    # Create and train the GRNN model
    model = GRNN(sigma=sigma)
    model.fit(X_train, y_train)
    
    # Return the model's validation accuracy
    return model.score(X_val, y_val)


def SVM_tune():
    param_ranges = {
    'C': [0.1, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
    }
    C, kernel, gamma = parameters
    
    # Convert the hyperparameter strings to the appropriate data types
    kernel = eval(kernel)
    gamma = eval(gamma)
    
    # Create and train the SVM model
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    
    # Return the model's validation accuracy
    return model.score(X_val, y_val)


def getFunctionDetails(a):
    param = {
        "F1": ["F1", -100, 100, 30],
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