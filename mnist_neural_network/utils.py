import numpy as np
import pandas as pd


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def ReLU_derivative(Z):
    return Z > 0


def one_hot_encoding(Y):
    E = np.zeros((58999, 10))
    for idx, y in enumerate(Y):
        E[idx][y] = 1
    
    return E.T  


def load_training_dataset(path):
    data = pd.read_csv(path)
    data = np.array(data)
    np.random.shuffle(data)
    m,n = data.shape

    data_validation = data[0:1000]
    Y_validation = data_validation[:, 0]
    Y_validation.shape = (1000, 1)
    X_validation = data_validation[:, 1:n]
    X_validation = X_validation / 255.

    data_train = data[1000:m]
    Y_train = data_train[:, 0]
    Y_train.shape = (m-1000, 1)
    X_train = data_train[:, 1:n]
    X_train = X_train / 255.

    return X_train, Y_train, X_validation, Y_validation


