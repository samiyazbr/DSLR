# DSLR Logistic Regression

## Setup and Installation

Before running the program, make sure to install the required dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```

## How to Run the Training Program

The logistic regression training program requires three arguments:
1. Path to the dataset CSV file
2. Name of the column containing the class labels
3. Output file path for the trained model (JSON format)

### Example Usage:

```bash
# Assuming your dataset is in the datasets folder
python logreg_train.py datasets/dataset_train.csv Hogwarts\ House model.json
```

This command will:
- Load the training data from `datasets/dataset_train.csv`
- Use the "Hogwarts House" column as the target class
- Save the trained model to `model.json`

### Notes:
- The program automatically detects and uses only the numerical features
- Features are normalized before training
- Training progress is displayed every 100 iterations
