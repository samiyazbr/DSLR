import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import load_data

def create_pair_plot(data, feature_names, class_column=None, max_features=None):
    """
    Create a pair plot (grid of scatter plots) for numerical features
    If class_column is specified, color points by class
    """
    # Convert data to numpy array
    data_array = np.array(data, dtype=object)
    
    # Find numerical columns
    numerical_indices = []
    numerical_features = []
    
    for i, col_name in enumerate(feature_names):
        if i == class_column:
            continue
            
        col = data_array[:, i]
        try:
            numeric_col = [float(x) for x in col if x != '' and not (isinstance(x, str) and x.strip() == '')]
            if len(numeric_col) > 0:
                numerical_indices.append(i)
                numerical_features.append(col_name)
        except (ValueError, TypeError):
            continue
    
    # Limit the number of features if needed
    if max_features and len(numerical_indices) > max_features:
        numerical_indices = numerical_indices[:max_features]
        numerical_features = numerical_features[:max_features]
    
    n_features = len(numerical_indices)
    
    # Extract numerical data
    numerical_data = np.zeros((len(data_array), n_features))
    for i, idx in enumerate(numerical_indices):
        for j, val in enumerate(data_array[:, idx]):
            try:
                if val != '' and not (isinstance(val, str) and val.strip() == ''):
                    numerical_data[j, i] = float(val)
                else:
                    numerical_data[j, i] = np.nan
            except (ValueError, TypeError):
                numerical_data[j, i] = np.nan
    
    # Extract class data if provided
    class_values = None
    if class_column is not None:
        class_values = np.array([str(x) for x in data_array[:, class_column]])
        unique_classes = list(set(class_values))
        unique_classes.sort()
    
    # Create the pair plot
    fig, axes = plt.subplots(n_features, n_features, figsize=(n_features * 3, n_features * 3))
    
    # Handle case of only one feature
    if n_features == 1:
        axes = np.array([[axes]])
    
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            # Create diagonal histograms
            if i == j:
                # Remove NaN values
                valid_indices = ~np.isnan(numerical_data[:, i])
                values = numerical_data[valid_indices, i]
                
                if class_column is None:
                    ax.hist(values, bins=20, alpha=0.7)
                else:
                    for cls in unique_classes:
                        class_indices = (class_values == cls) & valid_indices
                        if np.any(class_indices):
                            ax.hist(numerical_data[class_indices, i], bins=20, alpha=0.5, label=cls)
            
            # Create off-diagonal scatter plots
            else:
                # Remove rows with NaN values in either feature
                valid_indices = ~np.isnan(numerical_data[:, i]) & ~np.isnan(numerical_data[:, j])
                x = numerical_data[valid_indices, j]  # x-axis is column j
                y = numerical_data[valid_indices, i]  # y-axis is column i
                
                if class_column is None:
                    ax.scatter(x, y, alpha=0.5, s=10)
                else:
                    for cls in unique_classes:
                        class_indices = (class_values == cls) & valid_indices
                        if np.any(class_indices):
                            ax.scatter(numerical_data[class_indices, j], 
                                      numerical_data[class_indices, i], 
                                      alpha=0.5, s=10, label=cls)
            
            # Only add labels on the edges
            if i == n_features - 1:  # Bottom row
                ax.set_xlabel(numerical_features[j])
            if j == 0:  # Leftmost column
                ax.set_ylabel(numerical_features[i])
            
            # Remove tick labels for inner plots to avoid clutter
            if i < n_features - 1:
                ax.set_xticks([])
            if j > 0:
                ax.set_yticks([])
    
    # Add legend only once (in the top right subplot)
    if class_column is not None:
        axes[0, -1].legend(title=feature_names[class_column], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('pair_plot.png', bbox_inches='tight')
    plt.show()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python pair_plot.py <datafile.csv> [class_column_name] [max_features]")
        sys.exit(1)
        
    filepath = sys.argv[1]
    class_column = None
    max_features = None
    
    try:
        header, data = load_data(filepath)
        
        if len(sys.argv) >= 3:
            class_column_name = sys.argv[2]
            if class_column_name in header:
                class_column = header.index(class_column_name)
            else:
                print(f"Warning: Class column '{class_column_name}' not found in data")
        
        if len(sys.argv) == 4:
            max_features = int(sys.argv[3])
        
        create_pair_plot(data, header, class_column, max_features)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
