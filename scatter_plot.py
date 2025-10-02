import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import load_data

def create_scatter_plot(data, feature_names, x_feature, y_feature, class_column=None):
    """
    Create a scatter plot between two features
    If class_column is specified, color points by class
    """
    # Convert data to numpy array
    data_array = np.array(data, dtype=object)
    
    # Get column indices
    try:
        x_idx = feature_names.index(x_feature)
        y_idx = feature_names.index(y_feature)
    except ValueError:
        print(f"Error: One or both features not found in the data")
        return
    
    # Extract data points
    x_values = []
    y_values = []
    classes = []
    
    for i, row in enumerate(data_array):
        try:
            x_val = float(row[x_idx])
            y_val = float(row[y_idx])
            x_values.append(x_val)
            y_values.append(y_val)
            
            if class_column is not None:
                classes.append(row[class_column])
        except (ValueError, TypeError):
            continue
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    if class_column is None:
        plt.scatter(x_values, y_values, alpha=0.6)
    else:
        # Plot points colored by class
        unique_classes = list(set(classes))
        unique_classes.sort()
        
        for cls in unique_classes:
            indices = [i for i, c in enumerate(classes) if c == cls]
            plt.scatter([x_values[i] for i in indices], 
                        [y_values[i] for i in indices], 
                        label=cls, alpha=0.6)
        
        plt.legend(title=feature_names[class_column])
    
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"Scatter Plot: {y_feature} vs {x_feature}")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'scatter_{x_feature}_vs_{y_feature}.png')
    plt.show()

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python scatter_plot.py <datafile.csv> <x_feature> <y_feature> [class_column_name]")
        sys.exit(1)
        
    filepath = sys.argv[1]
    x_feature = sys.argv[2]
    y_feature = sys.argv[3]
    class_column = None
    
    try:
        header, data = load_data(filepath)
        
        if len(sys.argv) == 5:
            class_column_name = sys.argv[4]
            if class_column_name in header:
                class_column = header.index(class_column_name)
            else:
                print(f"Warning: Class column '{class_column_name}' not found in data")
        
        create_scatter_plot(data, header, x_feature, y_feature, class_column)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
