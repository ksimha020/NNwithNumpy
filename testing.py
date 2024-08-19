import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

W1 = np.load('W1.npy')
W2 = np.load('W2.npy')

b1 = np.load('b1.npy')
b2 = np.load('b2.npy')

data = np.array(pd.read_csv('test.csv'))

def getPrediction(A2):
    return np.argmax(A2, axis=1)

def relu_activation(Z):
    return np.maximum(0, Z)

def predict(index):
    pixels = data[index]/255
    predicted = (relu_activation(pixels.dot(W1)+b1)).dot(W2)+b2
    predicted = getPrediction(predicted)
    pixels = pixels.reshape(28, 28)
    print(predicted)
    plt.imshow(pixels, )
    plt.show()
    
    

predict(random.randint(0, 10000))