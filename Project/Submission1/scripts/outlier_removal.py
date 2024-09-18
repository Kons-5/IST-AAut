# outlier_removal.py
import numpy as np

def remove_outliers_iqr(y):
    """
    Removes outliers from the input array using the Interquartile Range (IQR) method.

    Parameters:
    y (numpy array): 1D array from which to remove outliers

    Returns:
    numpy array: Boolean mask indicating non-outlier entries
    """
    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create and return a boolean mask for the non-outliers
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
    from scipy import stats
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < threshold
    return mask
