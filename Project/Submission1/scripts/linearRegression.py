#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from outlier_removal import remove_outliers_iqr, remove_outliers_isolation_forest, remove_outliers_lof

# Load data
X_train = np.load("../data/X_train.npy")
y_train = np.load("../data/y_train.npy")
X_test = np.load("../data/X_test.npy")

# Data visualization
plt.figure(1)
plt.title("Boxplot of y_train (With Outliers)")
plt.boxplot(y_train, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))

# Dependent variable outlier removal
mask = remove_outliers_lof(y_train)
X_train_filtered = X_train[mask]
y_train_filtered = y_train[mask]

print(len(y_train_filtered))

plt.figure(2)
plt.title("Inliers detected")
plt.scatter(range(len(y_train)), y_train, c=mask, cmap='coolwarm', label='Inliers')
plt.xlabel("Sample index"); plt.ylabel("y values (Toxic Algae Concentration)"); plt.legend()

plt.figure(3)
plt.title("Boxplot of Filtered y_train (Outliers Removed)")
plt.boxplot(y_train_filtered, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.show()

# Apply regression model
model = LinearRegression()
model.fit(X_train_filtered, y_train_filtered)

# Predict values
y_pred = model.predict(X_train_filtered)

# Display the intercept and coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}\n")

# Calculate and print metrics
sse = np.sum((y_train_filtered - y_pred) ** 2)
mse = mean_squared_error(y_train_filtered, y_pred)
r2 = r2_score(y_train_filtered, y_pred)

print(f"SSE: {sse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.2f}")
