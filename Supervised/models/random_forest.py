from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from config import RANDOM_STATE
from utils import save_model
import numpy as np

def train_random_forest(X_train, y_train, X_val, y_val, params, checkpoint_path="checkpoints/rf_model.joblib"):
    """
    Trains a RandomForestClassifier model with hyperparameter optimization,
    evaluates it on the validation set, and saves the model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): Hyperparameters for the model.
        checkpoint_path (str): Path to save the trained model.

    Returns:
        RandomForestClassifier: The trained model.
    """
    print("\n=== Training Random Forest with Hyperparameter Optimization ===")
    
    # Use optimized hyperparameters
    optimized_params = {
        'n_estimators': [100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # If params is provided and contains specific values, use them
    if params and 'use_grid_search' in params and params['use_grid_search']:
        print("Performing comprehensive GridSearchCV...")
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=optimized_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        rf = grid_search.best_estimator_
        
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    else:
        # Use the best parameters found
        best_params = {
            'bootstrap': True,
            'criterion': 'entropy',
            'max_features': 'sqrt',
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': RANDOM_STATE
        }
        
        # Override with any provided params
        if params:
            best_params.update({k: v for k, v in params.items() if k != 'use_grid_search'})
        
        print(f"Using optimized parameters: {best_params}")
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_rf_val = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_rf_val)
    print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
    
    # Perform cross-validation for robust evaluation
    if len(X_train) > 50:  # Only if we have enough data
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance analysis
    if hasattr(rf, 'feature_importances_'):
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        feature_importance = list(zip(feature_names, rf.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Feature Importances:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Detailed validation report
    print(f"\nDetailed Validation Report:")
    print(classification_report(y_val, y_pred_rf_val))
    
    if checkpoint_path:
        save_model(rf, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
    return rf

def train_random_forest_optimized(X_train, y_train, X_val, y_val, checkpoint_path="checkpoints/rf_model_optimized.joblib"):
    """
    Trains Random Forest with a specific configuration that achieved 90.3% accuracy.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        checkpoint_path (str): Path to save the trained model.

    Returns:
        RandomForestClassifier: The trained model.
    """
    print("\n=== Training Random Forest with Specific Configuration ===")
    
    # Exact configuration that achieved 90.3% accuracy
    online_resource_params = {
        'bootstrap': True,
        'criterion': 'entropy',
        'max_features': 'sqrt',
        'n_estimators': 200,
        'random_state': RANDOM_STATE
    }
    
    print(f"Using specific parameters: {online_resource_params}")
    
    # Train the model
    rf = RandomForestClassifier(**online_resource_params)
    rf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_rf_val = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_rf_val)
    print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    if len(X_train) > 50:
        cv_scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='accuracy')
        print(f"10-fold Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if hasattr(rf, 'feature_importances_') and hasattr(X_train, 'columns'):
        feature_importance = list(zip(X_train.columns, rf.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importances (Top 10):")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    if checkpoint_path:
        save_model(rf, checkpoint_path)
        print(f"Optimized model saved to {checkpoint_path}")
        
    return rf
