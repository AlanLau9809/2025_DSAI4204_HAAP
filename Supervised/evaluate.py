import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd # Added for feature importance plotting
import numpy as np

def plot_feature_importance(model, feature_names, ax):
    """
    Plots feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        # Create a DataFrame for easier plotting
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax)
        ax.set_title('Feature Importance Ranking')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
    else:
        ax.set_title('Feature Importance Not Available')
        ax.text(0.5, 0.5, 'Model does not have feature_importances_ attribute.', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

def final_report(model, X_test, y_test, feature_names):
    """
    Generates and prints the final evaluation report for the model using the test set.
    This should only be called once at the end for final performance reporting.

    Args:
        model: The trained model to evaluate.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        feature_names (list): List of feature names for plotting.
    """
    y_final_pred = model.predict(X_test)
    
    # Get prediction probabilities for ROC curve
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)

    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*60)
    print("NOTE: This is the final evaluation - test set used only once!")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_final_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nFINAL RESULTS:")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_final_pred))

    # Create visualization plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance
    rf_model = None
    for name, estimator in model.estimators_:
        if name == 'rf': # Assuming 'rf' is the name for RandomForestClassifier
            rf_model = estimator
            break
    
    if rf_model:
        plot_feature_importance(rf_model, feature_names, ax1)
    else:
        ax1.set_title('Random Forest Model Not Found')
        ax1.text(0.5, 0.5, 'Could not find Random Forest estimator in ensemble.', 
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_final_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2) # Corrected ax=ax2
    ax2.set_title("Confusion Matrix")
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend(loc="lower right")
    ax3.grid(True)
    
    # 4. Performance Metrics Bar Chart
    metrics = ['Accuracy', 'ROC-AUC']
    values = [accuracy, roc_auc]
    colors = ['skyblue', 'lightcoral']
    bars = ax4.bar(metrics, values, color=colors)
    ax4.set_ylim([0, 1])
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics')
    ax4.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy, roc_auc

def plot_correlation_matrix(df):
    """
    Generates and displays the correlation matrix.

    Args:
        df (pd.DataFrame): The DataFrame for which to plot the correlation matrix.
    """
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    pass
