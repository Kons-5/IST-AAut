#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print(f"X_train shape: {X_train.shape}\n X_train type: {type(X_train)}\n")
print(f"y_train shape: {y_train.shape}\n y_train type: {type(y_train)}")
print(f"X_test shape: {X_test.shape}\n X_test type: {type(X_test)}\n")
print(f"y_test shape: {y_test.shape}\n y_test type: {type(y_test)}")

# Apply regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict values
y_pred = reg.predict(X)

# Coefficients and metrics
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Data visualization
"""
plt.figure(1), plt.scatter(X_train[:,0], y_train), plt.title("X1")
plt.figure(2), plt.scatter(X_train[:,1], y_train), plt.title("X2")
plt.figure(3), plt.scatter(X_train[:,2], y_train), plt.title("X3")
plt.figure(4), plt.scatter(X_train[:,3], y_train), plt.title("X4")
plt.figure(5), plt.scatter(X_train[:,4], y_train), plt.title("X5")
plt.show()
"""
