import numpy as np
import sys
import json
from utils import load_data, normalize_features, sigmoid, custom_mean

def load_model(model_file):
    """Load model parameters from file"""
    with open(model_file, 'r') as f:
        model = json.load(f)
    
    theta = np.array(model['theta'])
    classes = model['classes']
    features = model['features']
    
    return theta, classes, features

def predict(X, theta, classes):
    """Make predictions using trained logistic regression model"""
    # Add bias term
    X_with_bias = np.column_stack((np.ones(X.shape[0]), X))
    
    # Calculate probabilities for each class
    probabilities = sigmoid(np.dot(X_with_bias, theta.T))
    
    # Predict the class with highest probability
    pred_indices = np.argmax(probabilities, axis=1)
    predictions = [classes[idx] for idx in pred_indices]
    
    return np.array(predictions)

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python logreg_predict.py <model_file.json> <test_data.csv> [true_class_column]")
        sys.exit(1)
    
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    true_class_column = None
    
    if len(sys.argv) == 4:
        true_class_column = sys.argv[3]
    
    try:
        # Load model
        theta, classes, feature_names = load_model(model_file)
        
        # Load test data
        header, data = load_data(test_file)
        
        # Extract features from test data
        data_array = np.array(data, dtype=object)
        X_test = []
        
        # Find indices of required features in the test data
        feature_indices = []
        for feature in feature_names:
            if feature in header:
                feature_indices.append(header.index(feature))
            else:
                raise ValueError(f"Feature '{feature}' not found in test data")
        
        # Extract feature values
        for i in range(len(data_array)):
            try:
                row_features = [float(data_array[i, j]) for j in feature_indices]
                X_test.append(row_features)
            except (ValueError, TypeError):
                print(f"Warning: Skipping row {i+1} due to invalid data")
                X_test.append([np.nan] * len(feature_indices))
        
        X_test = np.array(X_test)
        
        # Handle missing values
        for col in range(X_test.shape[1]):
            col_values = X_test[:, col].tolist()
            col_mean = custom_mean(col_values)
            # Replace NaNs manually
            for i in range(X_test.shape[0]):
                if np.isnan(X_test[i, col]):
                    X_test[i, col] = col_mean
        
        # Normalize features
        X_test_normalized = normalize_features(X_test)
        
        # Make predictions
        y_pred = predict(X_test_normalized, theta, classes)
        
        # Output predictions
        output_file = 'houses.csv'
        with open(output_file, 'w') as f:
            f.write("Index,Hogwarts House\n")
            for i, prediction in enumerate(y_pred):
                f.write(f"{i},{prediction}\n")
        
        print(f"Predictions saved to {output_file}")
        
        # Calculate accuracy if true class column is provided
        if true_class_column:
            if true_class_column in header:
                true_idx = header.index(true_class_column)
                y_true = [data_array[i, true_idx] for i in range(len(data_array))]
                # Manual accuracy: proportion of exact matches
                correct = 0
                total = len(y_pred)
                for i in range(total):
                    if str(y_true[i]) == str(y_pred[i]):
                        correct += 1
                accuracy = correct / total if total > 0 else 0.0
                print(f"Accuracy: {accuracy:.4f}")
            else:
                print(f"Warning: True class column '{true_class_column}' not found in data")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
