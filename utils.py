import numpy as np
import csv
import pandas as pd

def load_data(filepath):
    """
    Load data from a CSV file and return header and data
    """
    with open(filepath, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = list(csv_reader)
    return header, data

def normalize_features(X):
    """
    Normalize features using min-max normalization
    """
    X_normalized = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[1]):
        col = X[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        
        # Avoid division by zero
        if max_val > min_val:
            X_normalized[:, i] = (col - min_val) / (max_val - min_val)
        else:
            X_normalized[:, i] = 0
    
    return X_normalized

def sigmoid(z):
    """
    Compute sigmoid function
    """
    return 1 / (1 + np.exp(-z))

def custom_count(values):
    """Calculate the count of non-NaN values."""
    return sum(~np.isnan(values))

def custom_mean(values):
    """Calculate the mean of values, ignoring NaN."""
    values = [x for x in values if not np.isnan(x)]
    if not values:
        return np.nan
    return sum(values) / len(values)

def custom_std(values):
    """Calculate the standard deviation, ignoring NaN."""
    values = [x for x in values if not np.isnan(x)]
    if not values or len(values) == 1:
        return np.nan
    mean = custom_mean(values)
    squared_diff = [(x - mean) ** 2 for x in values]
    return (sum(squared_diff) / (len(values) - 1)) ** 0.5

def custom_min(values):
    """Find the minimum value, ignoring NaN."""
    values = [x for x in values if not np.isnan(x)]
    if not values:
        return np.nan
    return min(values)

def custom_max(values):
    """Find the maximum value, ignoring NaN."""
    values = [x for x in values if not np.isnan(x)]
    if not values:
        return np.nan
    return max(values)

def custom_percentile(values, percentile):
    """Calculate the specified percentile, ignoring NaN."""
    values = [x for x in values if not np.isnan(x)]
    if not values:
        return np.nan
    
    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate the index position
    idx = (n - 1) * percentile / 100
    
    if idx.is_integer():
        return sorted_values[int(idx)]
    else:
        # Interpolate between the two nearest elements
        lower_idx = int(np.floor(idx))
        upper_idx = int(np.ceil(idx))
        lower_value = sorted_values[lower_idx]
        upper_value = sorted_values[upper_idx]
        fraction = idx - lower_idx
        return lower_value + fraction * (upper_value - lower_value)
