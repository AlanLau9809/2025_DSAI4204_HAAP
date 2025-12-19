from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from config import RANDOM_STATE
from utils import save_model

def train_gradient_boosting(X_train, y_train, X_val, y_val, params, checkpoint_path="checkpoints/gb_model.joblib"):
    """
    Trains a GradientBoostingClassifier model, evaluates it on the validation set, and saves the model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict): Hyperparameters for the model.
        checkpoint_path (str): Path to save the trained model.

    Returns:
        GradientBoostingClassifier: The trained model.
    """
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    gb.fit(X_train, y_train)
    
    y_pred_gb_val = gb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_gb_val)
    print(f"Gradient Boosting Validation Accuracy: {accuracy:.4f}")
    
    if checkpoint_path:
        save_model(gb, checkpoint_path)
        
    return gb
