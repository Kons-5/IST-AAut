#!/bin/python3

import lof_tuner
import numpy as np
from lof_tuner import LOF_AutoTuner
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

# Load data
X_train = np.load("../../data/X_train.npy")
y_train = np.load("../../data/y_train.npy")

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
residuals = (y_train - y_train_pred)**2
data = residuals.reshape(-1, 1)


# Tune eps and min_samples based on your data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data)

# Get the cluster labels
labels = dbscan.labels_

# Count the number of points in each cluster
unique, counts = np.unique(labels, return_counts=True)
cluster_info = dict(zip(unique, counts))

# Print the cluster information
print("Cluster information (label: size):", cluster_info)
