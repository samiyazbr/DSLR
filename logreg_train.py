import numpy as np
import sys
import json
from utils import load_data, normalize_features, sigmoid

def train_one_vs_all(X, y, classes, learning_rate=0.1, num_iterations=1000):
    """
    Train logistic regression model using one-vs-all approach
    """
    n_samples, n_features = X.shape
    n_classes = len(classes)
    
    # Initialize weights and bias for each class
    all_theta = np.zeros((n_classes, n_features + 1))
    
    # Add bias term to X
    X_with_bias = np.column_stack((np.ones(n_samples), X))
    
    # Train logistic regression for each class
    for i, class_val in enumerate(classes):
        print(f"Training for class {class_val} ({i+1}/{n_classes})...")
        
        # Create binary labels (1 for current class, 0 for others)
        binary_y = np.array([1 if val == class_val else 0 for val in y])
        
        # Initialize parameters
        theta = np.zeros(n_features + 1)
        
        # Gradient descent
        for iteration in range(num_iterations):
            # Calculate hypothesis
            z = np.dot(X_with_bias, theta)
            h = sigmoid(z)
            
            # Calculate gradient
            gradient = np.dot(X_with_bias.T, (h - binary_y)) / n_samples
            
            # Update parameters
            theta = theta - learning_rate * gradient
            
            # Log progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                # Calculate cost function
                epsilon = 1e-5  # To prevent log(0)
                h_clipped = np.clip(h, epsilon, 1 - epsilon)
                cost = -np.mean(binary_y * np.log(h_clipped) + (1 - binary_y) * np.log(1 - h_clipped))
                print(f"Iteration {iteration + 1}/{num_iterations}, Cost: {cost:.6f}")
        
        all_theta[i] = theta
    
    return all_theta, classes

def save_model(theta, classes, features, output_file):
    """Save the trained model parameters to a file"""
    model = {
        'theta': theta.tolist(),
        'classes': classes.tolist() if isinstance(classes, np.ndarray) else classes,
        'features': features
    }
    
    with open(output_file, 'w') as f:
        json.dump(model, f)
    
    print(f"Model saved to {output_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python logreg_train.py <datafile.csv> <class_column_name> <output_model.json>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    class_column_name = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        # Load data
        header, data = load_data(filepath)
        data_array = np.array(data, dtype=object)
        
        # Find class column
        if class_column_name not in header:
            raise ValueError(f"Class column '{class_column_name}' not found in data")
        
        class_idx = header.index(class_column_name)
        
        # Extract features and target
        feature_names = [name for i, name in enumerate(header) if i != class_idx]
        
        # Find numerical columns for features
        X = []
        y = []
        valid_feature_indices = []
        valid_feature_names = []
        
        # Extract class values
        y = [row[class_idx] for row in data_array]
        
        # Extract valid numerical features
        for i, name in enumerate(header):
            if i != class_idx:
                try:
                    # Try to convert the column to float
                    values = [float(row[i]) for row in data_array]
                    # Check if there are any NaN values
                    if not any(np.isnan(values)):
                        X.append(values)
                        valid_feature_indices.append(i)
                        valid_feature_names.append(name)
                except (ValueError, TypeError):
                    continue
        
        # Convert to numpy arrays
        X = np.array(X).T  # Transpose to get samples as rows
        y = np.array(y)
        
        # Normalize features
        X_normalized = normalize_features(X)
        
        # Get unique classes
        classes = np.unique(y)
        
        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Classes: {classes}")
        
        # Train logistic regression
        theta, trained_classes = train_one_vs_all(X_normalized, y, classes)
        
        # Save model
        save_model(theta, trained_classes, valid_feature_names, output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
