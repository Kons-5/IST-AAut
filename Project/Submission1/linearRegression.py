#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")

# Apply regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_train)

# Coefficients and metrics
print("Coefficients: \n", model.coef_)
print("SSE: %.2f" % np.sum((y_train - y_pred)**2))
print("MSE: %.2f" % mean_squared_error(y_train, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_train, y_pred))

# Data visualization
plt.boxplot(y_train, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.show()

"""
plt.figure(1), plt.scatter(X_train[:,0], y_train), plt.title("X1")
plt.figure(2), plt.scatter(X_train[:,1], y_train), plt.title("X2")
plt.figure(3), plt.scatter(X_train[:,2], y_train), plt.title("X3")
plt.figure(4), plt.scatter(X_train[:,3], y_train), plt.title("X4")
plt.figure(5), plt.scatter(X_train[:,4], y_train), plt.title("X5")
plt.show()
"""
