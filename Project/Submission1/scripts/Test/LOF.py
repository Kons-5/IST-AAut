#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.legend_handler import HandlerPathCollection


# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Apply LOF to y_train
lof = LocalOutlierFactor(n_neighbors=98, contamination=0.25)
y_pred_outliers = lof.fit_predict(y_train.reshape(-1, 1))
X_scores = lof.negative_outlier_factor_

# Identify outliers
outlier_mask = y_pred_outliers == -1  # Mask for the outliers
inlier_mask = y_pred_outliers == 1    # Mask for the inliers

# Filter the inliers for X_train and y_train
X_train_cleaned = X_train[inlier_mask]
y_train_cleaned = y_train[inlier_mask]

# Custom function for marker size in the legend


def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


print(f"Size of data set after outlier removal: {len(y_train_cleaned)}")
plt.scatter(range(len(y_train)), y_train, c=y_pred_outliers,
            cmap='coolwarm', s=30, label="Data points")
plt.title("Outliers detected by LOF")
plt.xlabel("Sample index")
plt.ylabel("y_train (Toxic Algae Concentration)")

# Plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    range(len(y_train)),
    y_train,
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

# Customize the plot
plt.xlabel("Sample index")
plt.ylabel("y_train (Toxic Algae Concentration)")
plt.title("Local Outlier Factor (LOF) - Outlier Scores")
plt.legend(handler_map={scatter: HandlerPathCollection(
    update_func=update_legend_marker_size)})

plt.tight_layout()
plt.show()
