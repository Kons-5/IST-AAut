# outlier_removal.py

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

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

def remove_outliers_zscore(y, threshold=3):
    """
    Removes outliers from the input array using the Z-score method.

    Parameters:
    y (numpy array): 1D array from which to remove outliers
    threshold (float): Z-score threshold for identifying outliers (default is 3)

    Returns:
    numpy array: Boolean mask indicating non-outlier entries
    """
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < threshold
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
