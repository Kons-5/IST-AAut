#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt

trainX = np.load("X_train.npy")
trainY = np.load("y_train.npy")

print(f"X_train shape: {trainX.shape}; Type: {type(trainX)}")
print(f"y_train shape: {trainY.shape}; Type: {type(trainY)}")

plt.figure(1), plt.scatter(trainX[:,0], trainY), plt.title("X1")
plt.figure(2), plt.scatter(trainX[:,1], trainY), plt.title("X2")
plt.figure(3), plt.scatter(trainX[:,2], trainY), plt.title("X3")
plt.figure(4), plt.scatter(trainX[:,3], trainY), plt.title("X4")
plt.figure(5), plt.scatter(trainX[:,4], trainY), plt.title("X5")
plt.show()
