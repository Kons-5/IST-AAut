# outlier_removal.py

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def remove_outliers_iqr(y):
    """
    Removes outliers from the input array using the IQR method.

    Parameters:
    y (numpy array): 1D array from which to remove outliers

    Returns:
    numpy array: Boolean mask indicating non-outlier entries
    """
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (y >= lower_bound) & (y <= upper_bound)
    return mask

def remove_outliers_isolation_forest(y, contamination=0.25):
    """
    Removes outliers from the input array using the Isolation Forest method.

    Parameters:
    y (numpy array): 1D array from which to remove outliers
    contamination (float): The proportion of outliers in the data (default is 0.25)

    Returns:
    numpy array: Boolean mask indicating non-outlier entries
    """
    isolation_forest = IsolationForest(contamination=contamination, random_state=1)
    outlier_labels = isolation_forest.fit_predict(y.reshape(-1, 1))  # Reshape y to 2D
    mask = outlier_labels == 1  # Inliers are labeled as 1, outliers as -1
    return mask


def remove_outliers_lof(y, contamination=0.25, n_neighbors=60):
    """
    Removes outliers from the input array using the Local Outlier Factor (LOF) method.

    Parameters:
    y (numpy array): 1D array from which to remove outliers
    contamination (float): The proportion of outliers in the data (default is 0.25)
    n_neighbors (int): Number of neighbors to use for LOF

    Returns:
    numpy array: Boolean mask indicating non-outlier entries
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(y.reshape(-1, 1))
    mask = outlier_labels == 1
    return mask
