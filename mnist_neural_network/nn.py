import matplotlib.pyplot as plt
import numpy as np
import utils


def initalise_paramters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def feedforward(X, W1, b1, W2, b2):
    S1 = np.dot(W1, X.T) + b1
    A1 = utils.ReLU(S1)
    S2 = np.dot(W2, A1) + b2
    A2 = utils.softmax(S2)

    return S1, A1, S2, A2


def backward_propagation(X, Y,S1, A1, A2, W2):
    num_examples = len(Y)
    encodingY = utils.one_hot_encoding(Y)

    errorOutputLayer = A2 - encodingY
    derivW2 = np.dot(errorOutputLayer, A1.T) / num_examples
    derivB2 = np.sum(errorOutputLayer) / num_examples


    errorHiddenLayer = np.multiply(np.dot(W2.T, errorOutputLayer), utils.ReLU_derivative(S1))
    derivW1 = np.dot(errorHiddenLayer, X) / num_examples
    derivB1 = np.sum(errorHiddenLayer) / num_examples

    return derivW1, derivB1, derivW2, derivB2


def update_parameters(derivW1, derivB1, derivW2, derivB2, W1, b1, W2, b2, alpha):
    W1_new = W1 - alpha * derivW1
    b1_new = b1 - alpha * derivB1 
    W2_new = W2 - alpha * derivW2
    b2_new = b2 - alpha * derivB2

    return W1_new, b1_new, W2_new, b2_new


def train(X, Y, alpha, epochs):
    W1, b1, W2, b2 = initalise_paramters()
    for i in range(1, epochs+1):
        print(f'Epoch: {i}')
        S1, A1, S2, A2 = feedforward(X, W1, b1, W2, b2)
        derivW1, derivB1, derivW2, derivB2 = backward_propagation(X, Y, S1, A1, A2, W2)
        W1, b1, W2, b2 = update_parameters(derivW1, derivB1, derivW2, derivB2, W1, b1, W2, b2, alpha)

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    X.shape = (1, 784)
    S1 = np.dot(W1, X.T) + b1
    A1 = utils.ReLU(S1)
    S2 = np.dot(W2, A1) + b2
    A2 = utils.softmax(S2)
    return A2.argmax()


def compute_accuracy(X, Y, W1, b1, W2, b2):
    correct = 0
    for x, y in zip(X, Y):
        prediction = predict(x, W1, b1, W2, b2)
        if prediction == y:
            correct += 1

    return float(correct) / float(len(Y))


def test_prediction(X, Y, W1, b1, W2, b2):
    yhat = predict(X, W1, b1, W2, b2)
    print('Prediction: ', yhat, 'Actual: ', Y)



def __main__():
    X_train, Y_train, X_val, Y_val = utils.load_training_dataset('mnist_train.csv')
    W1, b1, W2, b2 = train(X_train,Y_train, 0.1, 500)
    accuracy = compute_accuracy(X_val, Y_val, W1, b1, W2, b2)
    print(f'Accuracy: {accuracy}')

    test_prediction(X_val[233], Y_val[233], W1, b1, W2, b2)
    test_prediction(X_val[11], Y_val[11], W1, b1, W2, b2)
    test_prediction(X_val[789], Y_val[789], W1, b1, W2, b2)
    test_prediction(X_val[10], Y_val[10], W1, b1, W2, b2)


if __name__ == '__main__':
    __main__()