# config.py

# File path for the dataset
DATA_PATH = "/Users/chunmanchan/Downloads/Alan/DSAI4204/archive/heart.csv"

# List of categorical and continuous feature names
CAT_COLS = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
CON_COLS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

# Target column name
TARGET_COL = 'output'

# Test size for splitting the data
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Random state for reproducibility
RANDOM_STATE = 42

# Model Hyperparameters - Optimized based on online resource
LR_PARAMS = {
    'penalty': 'l1', 
    'solver': 'saga', 
    'max_iter': 1000,
    'random_state': 42
}
SVM_PARAMS = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
RF_PARAMS = {
    'bootstrap': True,
    'criterion': 'entropy',
    'max_features': 'sqrt',  # 'auto' is deprecated, use 'sqrt' instead
    'n_estimators': 200,
    'random_state': 42
}
GB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

# Checkpoint paths
LR_CHECKPOINT = "checkpoints/lr_model.joblib"
SVM_CHECKPOINT = "checkpoints/svm_model.joblib"
RF_CHECKPOINT = "checkpoints/rf_model.joblib"
GB_CHECKPOINT = "checkpoints/gb_model.joblib"
VOTING_CHECKPOINT = "checkpoints/voting_model.joblib"
