import pandas as pd
import sys
import numpy as np

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
    min_val = None
    for x in values:
        if np.isnan(x):
            continue
        if min_val is None or x < min_val:
            min_val = x
    return np.nan if min_val is None else min_val

def custom_max(values):
    """Find the maximum value, ignoring NaN."""
    max_val = None
    for x in values:
        if np.isnan(x):
            continue
        if max_val is None or x > max_val:
            max_val = x
    return np.nan if max_val is None else max_val

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

def load_data(filepath):
    """Load data from CSV file and return a pandas DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def display_statistics(dataframe):
    """
    Calculate and display statistics for all numerical features in the dataframe.
    """
    # Select only numerical columns
    numerical_df = dataframe.select_dtypes(include=['number'])
    
    # Get column values as lists for custom calculations
    columns = numerical_df.columns
    values_by_column = {col: numerical_df[col].values.tolist() for col in columns}
    
    # Calculate statistics using custom functions
    count = [custom_count(values_by_column[col]) for col in columns]
    mean = [custom_mean(values_by_column[col]) for col in columns]
    std = [custom_std(values_by_column[col]) for col in columns]
    min_val = [custom_min(values_by_column[col]) for col in columns]
    q25 = [custom_percentile(values_by_column[col], 25) for col in columns]
    q50 = [custom_percentile(values_by_column[col], 50) for col in columns]
    q75 = [custom_percentile(values_by_column[col], 75) for col in columns]
    max_val = [custom_max(values_by_column[col]) for col in columns]
    
    # Create feature names
    feature_names = []
    for i in range(len(columns)):
        feature_names.append(f"Feature {i+1}")
    
    # Define formatting
    col_width = 15
    
    # Print headers
    header_row = "".ljust(10)
    for name in feature_names:
        header_row += name.ljust(col_width)
    print(header_row)
    
    # Print each statistics row with proper alignment
    stats = [
        ("Count", count),
        ("Mean", mean),
        ("Std", std),
        ("Min", min_val),
        ("25%", q25),
        ("50%", q50),
        ("75%", q75),
        ("Max", max_val)
    ]
    
    for name, values in stats:
        row = name.ljust(10)
        for val in values:
            row += f"{val:.6f}".ljust(col_width)
        print(row)

def main():
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file.csv>")
        sys.exit(1)
    
    try:
        # Load the dataset using the utility function
        file_path = sys.argv[1]
        data = load_data(file_path)
        
        if data is None:
            sys.exit(1)
            
        # Display statistics
        display_statistics(data)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
