#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Range of n_neighbors to try
n_neighbors_range = list(range(20, 200))

# Store the average LOF score for each n_neighbors
avg_lof_scores = []

for n in n_neighbors_range:
    lof = LocalOutlierFactor(n_neighbors=n, contamination=0.25)
    lof.fit(y_train.reshape(-1, 1))
    lof_scores = -lof.negative_outlier_factor_
    avg_lof_scores.append(np.mean(lof_scores))


index = max(avg_lof_scores)
index = avg_lof_scores.index(index)

# Plot the average LOF scores
print(
    f"Highest average LOF Score: {n_neighbors_range[index]}")
plt.plot(n_neighbors_range, avg_lof_scores, marker='.')
plt.title("Average LOF Score vs. Number of Neighbors")
plt.xlabel("Number of Neighbors (n_neighbors)")
plt.ylabel("Average LOF Score")
plt.show()
