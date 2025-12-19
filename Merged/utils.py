import joblib
import os

def save_model(model, filename):
    """Saves a model to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Loads a model from a file."""
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    return None
