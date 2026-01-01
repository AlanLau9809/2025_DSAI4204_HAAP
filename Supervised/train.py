from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from utils import load_model, save_model
from config import LR_PARAMS, SVM_PARAMS, RF_PARAMS, GB_PARAMS, LR_CHECKPOINT, SVM_CHECKPOINT, RF_CHECKPOINT, GB_CHECKPOINT, VOTING_CHECKPOINT
from models.logistic_regression import train_logistic_regression
from models.svm import train_svm
from models.random_forest import train_random_forest
from models.gradient_boosting import train_gradient_boosting

def train_ensemble_model(X_train, y_train, X_val, y_val, use_checkpoints=True):
    """
    Trains or loads individual models with optimized parameters and then trains a final ensemble model.
    This approach leverages optimized techniques while maintaining ensemble benefits.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        use_checkpoints (bool): Whether to load models from checkpoints if they exist.

    Returns:
        VotingClassifier: The trained final ensemble model.
    """
    
    print("\n" + "="*70)
    print("TRAINING OPTIMIZED ENSEMBLE MODEL")
    print("="*70)
    
    individual_scores = {}
    
    # --- Logistic Regression (Optimized) ---
    print("\nTraining Optimized Logistic Regression...")
    lr_model = None
    if use_checkpoints:
        lr_model = load_model(LR_CHECKPOINT)
    if lr_model is None:
        lr_model = train_logistic_regression(X_train, y_train, X_val, y_val, LR_PARAMS, LR_CHECKPOINT)
    
    # Evaluate individual model
    lr_pred = lr_model.predict(X_val)
    lr_score = accuracy_score(y_val, lr_pred)
    individual_scores['Logistic Regression'] = lr_score
    print(f"   Individual LR Accuracy: {lr_score:.4f}")

    # --- SVM ---
    print("\nTraining SVM...")
    svm_model = None
    if use_checkpoints:
        svm_model = load_model(SVM_CHECKPOINT)
    if svm_model is None:
        svm_model = train_svm(X_train, y_train, X_val, y_val, SVM_PARAMS, SVM_CHECKPOINT)
    
    # Evaluate individual model
    svm_pred = svm_model.predict(X_val)
    svm_score = accuracy_score(y_val, svm_pred)
    individual_scores['SVM'] = svm_score
    print(f"   Individual SVM Accuracy: {svm_score:.4f}")
        
    # --- Random Forest (Optimized) ---
    print("\nTraining Optimized Random Forest...")
    rf_model = None
    if use_checkpoints:
        rf_model = load_model(RF_CHECKPOINT)
    if rf_model is None:
        rf_model = train_random_forest(X_train, y_train, X_val, y_val, RF_PARAMS, RF_CHECKPOINT)
    
    # Evaluate individual model
    rf_pred = rf_model.predict(X_val)
    rf_score = accuracy_score(y_val, rf_pred)
    individual_scores['Random Forest'] = rf_score
    print(f"   Individual RF Accuracy: {rf_score:.4f}")

    # --- Gradient Boosting ---
    print("\nTraining Gradient Boosting...")
    gb_model = None
    if use_checkpoints:
        gb_model = load_model(GB_CHECKPOINT)
    if gb_model is None:
        gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val, GB_PARAMS, GB_CHECKPOINT)
    
    # Evaluate individual model
    gb_pred = gb_model.predict(X_val)
    gb_score = accuracy_score(y_val, gb_pred)
    individual_scores['Gradient Boosting'] = gb_score
    print(f"   Individual GB Accuracy: {gb_score:.4f}")

    # --- Individual Model Performance Summary ---
    print(f"\nINDIVIDUAL MODEL PERFORMANCE:")
    print("-" * 40)
    best_individual_score = 0
    best_individual_model = None
    for model_name, score in individual_scores.items():
        print(f"   {model_name:20s}: {score:.4f} ({score*100:.2f}%)")
        if score > best_individual_score:
            best_individual_score = score
            best_individual_model = model_name
    
    print(f"\nBest Individual Model: {best_individual_model} ({best_individual_score:.4f})")
    
    # --- Voting Classifier ---
    print(f"\nTraining Ensemble Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_model), 
            ('svm', svm_model), 
            ('rf', rf_model), 
            ('gb', gb_model)
        ], 
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)
    
    # Evaluate ensemble
    ensemble_pred = voting_clf.predict(X_val)
    ensemble_score = accuracy_score(y_val, ensemble_pred)
    print(f"   Ensemble Accuracy: {ensemble_score:.4f} ({ensemble_score*100:.2f}%)")
    
    # Compare ensemble vs best individual
    improvement = ensemble_score - best_individual_score
    if improvement > 0:
        print(f"   Ensemble Improvement: +{improvement:.4f} over best individual")
    else:
        print(f"   Ensemble vs Best Individual: {improvement:.4f}")
    
    # Save the final ensemble model
    save_model(voting_clf, VOTING_CHECKPOINT)
    print(f"\nEnsemble model saved to {VOTING_CHECKPOINT}")

    return voting_clf

def get_best_individual_model(X_train, y_train, X_val, y_val, use_checkpoints=True):
    """
    Train individual models and return the best performing one.
    Useful for comparison with ensemble approach.
    
    Returns:
        tuple: (best_model, model_name, score)
    """
    print("\nFinding Best Individual Model...")
    
    models = {}
    scores = {}
    
    # Train all models
    lr_model = load_model(LR_CHECKPOINT) if use_checkpoints else None
    if lr_model is None:
        lr_model = train_logistic_regression(X_train, y_train, X_val, y_val, LR_PARAMS, LR_CHECKPOINT)
    
    rf_model = load_model(RF_CHECKPOINT) if use_checkpoints else None
    if rf_model is None:
        rf_model = train_random_forest(X_train, y_train, X_val, y_val, RF_PARAMS, RF_CHECKPOINT)
    
    # Evaluate models
    models['Logistic Regression'] = lr_model
    models['Random Forest'] = rf_model
    
    for name, model in models.items():
        pred = model.predict(X_val)
        score = accuracy_score(y_val, pred)
        scores[name] = score
    
    # Find best
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    best_score = scores[best_name]
    
    print(f"Best individual model: {best_name} with {best_score:.4f} accuracy")
    
    return best_model, best_name, best_score
