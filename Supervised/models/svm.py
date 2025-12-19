from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from config import RANDOM_STATE
from utils import save_model

def train_svm(X_train, y_train, X_val, y_val, params, checkpoint_path="checkpoints/svm_model.joblib"):
    """
    Trains and tunes an SVM model using GridSearchCV, evaluates it, and saves the best model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): Hyperparameter grid for GridSearchCV.
        checkpoint_path (str): Path to save the trained model.

    Returns:
        SVC: The best trained SVM model.
    """
    print("\nTuning SVM with GridSearch on validation set...")
    
    # The 'refit' parameter makes GridSearchCV train a new model on the whole training set (X_train, y_train) 
    # with the best parameters found.
    grid = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), params, refit=True, verbose=0, cv=3) # Using 3-fold cross-validation on the training data
    grid.fit(X_train, y_train)
    
    best_svm = grid.best_estimator_
    
    y_pred_best_svm_val = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_best_svm_val)
    
    print(f"Best SVM Parameters: {grid.best_params_}")
    print(f"Best SVM Validation Accuracy: {accuracy:.4f}")
    
    if checkpoint_path:
        save_model(best_svm, checkpoint_path)
        
    return best_svm
