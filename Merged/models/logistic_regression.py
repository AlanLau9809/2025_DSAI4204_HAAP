from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from config import RANDOM_STATE
from utils import save_model
import warnings

def train_logistic_regression(X_train, y_train, X_val, y_val, params, checkpoint_path="checkpoints/lr_model.joblib"):
    """
    Trains a LogisticRegression model with optimized hyperparameters,
    evaluates it on the validation set, and saves the model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): Hyperparameters for the model.
        checkpoint_path (str): Path to save the trained model.

    Returns:
        LogisticRegression: The trained model.
    """
    print("\n=== Training Logistic Regression with Optimized Parameters ===")
    
    # Suppress convergence warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Use optimized parameters from online resource if available
    if params and 'use_grid_search' in params and params['use_grid_search']:
        print("Performing GridSearchCV for Logistic Regression...")
        
        # Parameter grid for optimization
        param_grid = {
            'penalty': ['l1', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': [0.1, 1, 10, 100],
            'max_iter': [1000, 2000]
        }
        
        # Create base model
        lr_base = LogisticRegression(random_state=RANDOM_STATE)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=lr_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        lr = grid_search.best_estimator_
        
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    else:
        # Use the optimized parameters from online resource
        optimized_params = {
            'penalty': 'l1',
            'solver': 'saga',
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        }
        
        # Override with any provided params
        if params:
            optimized_params.update({k: v for k, v in params.items() if k != 'use_grid_search'})
        
        print(f"Using optimized parameters: {optimized_params}")
        lr = LogisticRegression(**optimized_params)
        lr.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_lr_val = lr.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_lr_val)
    print(f"Logistic Regression Validation Accuracy: {accuracy:.4f}")
    
    # Cross-validation for robust evaluation
    if len(X_train) > 50:
        cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed validation report
    print(f"\nDetailed Validation Report:")
    print(classification_report(y_val, y_pred_lr_val))
    
    if checkpoint_path:
        save_model(lr, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
    return lr
