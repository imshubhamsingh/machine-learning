#   XOR.py-A very simple neural network to do exclusive or.
import numpy as np

epochs = 500  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])


def sigmoid(x): return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x): return x * (1 - x)  # derivative of sigmoid


# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
print("Original output layer weights: \n", Wz)
print("Original hidden layer weights: \n", Wh)

for i in range(epochs):
    H = sigmoid(np.dot(X, Wh))  # hidden layer results
    # print("Hidden", H)
    Z = sigmoid(np.dot(H, Wz))  # output layer results
    # print("Output: \n", Z)
    E = Y - Z  # how much we missed (error)
    # print("error: \n", E)
    dZ = E * sigmoid_(Z)  # delta Z
    # print("dz", dZ)
    # print("Wz.T", Wz.T)
    dH = dZ.dot(Wz.T) * sigmoid_(H)  # delta H
    # print("dH", dH)
    print("weight adjusted")
    Wz += H.T.dot(dZ)  # update output layer weights
    # print("H.T.dot(dZ)", H.T.dot(dZ))
    Wh += X.T.dot(dH)  # update hidden layer weights
    print("output layer weights: \n", Wz)
    print("hidden layer weights: \n", Wh)

print("Z: ", Z)  # what have we learnt?
