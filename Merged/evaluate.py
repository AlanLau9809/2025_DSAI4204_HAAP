import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np

def final_report(model, X_test, y_test):
    """
    Generates and prints the final evaluation report for the model using the test set.
    This should only be called once at the end for final performance reporting.

    Args:
        model: The trained model to evaluate.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
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
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_final_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True)
    
    # 3. Prediction Distribution
    ax3.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='No Heart Disease', color='blue')
    ax3.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Heart Disease', color='red')
    ax3.set_xlabel('Prediction Probability')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Probability Distribution')
    ax3.legend()
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
