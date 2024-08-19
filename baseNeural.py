import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#global variables
data = pd.read_csv('train.csv')
data = np.array(data)

np.random.shuffle(data)
X_data = data[:, 1:]/255
Y_data = data[:, 0]


X_train = X_data[1000:, :]
Y_train = Y_data[1000:]



def init_params(m):
    W1 = np.random.randn(m, 10) * np.sqrt(2. / m)
    W2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.zeros((1, 10))
    b1 = np.zeros((1, 10))

    return W2, W1, b2, b1

def relu_activation(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return (Z > 0).astype('float')

def softMax_activation(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

def forwardProp(W2, W1, b2, b1, X):
    Z1 = X.dot(W1) + b1
    A1 = relu_activation(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softMax_activation(Z2)

    return Z1, Z2, A1, A2

def oneHotEncoderY(Y):
    oneHot = np.zeros((Y.size, 10))
    oneHot[np.arange(Y.size), Y] = 1

    return oneHot

def gradient_calc(Z1, A1, A2, X, Y, W2):
    n = X.shape[0]
    oneHotY = oneHotEncoderY(Y)
    del1 = A2 - oneHotY
    del2 = del1.dot(W2.T) * relu_deriv(Z1)
    dW2 = (A1.T).dot(del1) / n
    db2 = np.sum(del1, axis=0, keepdims=True) / n
    dW1 = (X.T).dot(del2) / n
    db1 = np.sum(del2, axis=0, keepdims=True) / n

    return dW2, dW1, db2, db1

def backwardProp(Z1, A1, A2, X, Y, W2, W1, b2, b1, alpha):
    dW2, dW1, db2, db1 = gradient_calc(Z1, A1, A2, X, Y, W2)
    W2 = W2 - alpha*dW2
    W1 = W1 - alpha*dW1
    b2 = b2 - alpha*db2
    b1 = b1 - alpha*db1

    return W2, W1, b2, b1


def getPrediction(A2):
    return np.argmax(A2, axis=1)

def accuracyCheck(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y)/Y.size

def neuralNet(X, Y, alpha, epochs):
    W2, W1, b2, b1 = init_params(784)

    for i in range(epochs):
        Z1, Z2, A1, A2 = forwardProp(W2, W1, b2, b1, X)
        W2, W1, b2, b1 = backwardProp(Z1, A1, A2, X, Y, W2, W1, b2, b1, alpha)

        if(i % 10 == 0):
            predictions = getPrediction(A2)
            print(accuracyCheck(predictions, Y))
    
    return W2, W1, b2, b1

W2, W1, b2, b1 = neuralNet(X_train, Y_train, 0.1, 200)

np.save('W2.npy', W2)
np.save('W1.npy', W1)

np.save('b2.npy', b2)
np.save('b1.npy', b1)

