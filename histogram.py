import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import load_data

def create_histograms(data, feature_names, class_column=None):
    """
    Create histograms for all numerical features
    If class_column is specified, color histograms by class
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
    
    # Set up the figure grid
    n_features = len(numerical_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # If class column is specified, get class values
    class_values = None
    if class_column is not None:
        class_values = list(set(data_array[:, class_column]))
        class_values.sort()
    
    # Create histograms
    for i, (idx, feature) in enumerate(zip(numerical_indices, numerical_features)):
        ax = axes[i]
        
        if class_column is None:
            # Create single histogram
            values = [float(x) for x in data_array[:, idx] if x != '' and not (isinstance(x, str) and x.strip() == '')]
            ax.hist(values, bins=20, alpha=0.7)
        else:
            # Create separate histogram for each class
            for class_val in class_values:
                class_indices = [j for j, x in enumerate(data_array[:, class_column]) if x == class_val]
                values = [float(data_array[j, idx]) for j in class_indices 
                         if data_array[j, idx] != '' and not (isinstance(data_array[j, idx], str) and data_array[j, idx].strip() == '')]
                ax.hist(values, bins=20, alpha=0.5, label=str(class_val))
            
            ax.legend(title=feature_names[class_column])
        
        ax.set_title(feature)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    # Hide unused axes
    for i in range(len(numerical_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.show()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python histogram.py <datafile.csv> [class_column_name]")
        sys.exit(1)
        
    filepath = sys.argv[1]
    class_column = None
    
    try:
        header, data = load_data(filepath)
        
        if len(sys.argv) == 3:
            class_column_name = sys.argv[2]
            if class_column_name in header:
                class_column = header.index(class_column_name)
            else:
                print(f"Warning: Class column '{class_column_name}' not found in data")
        
        create_histograms(data, header, class_column)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
