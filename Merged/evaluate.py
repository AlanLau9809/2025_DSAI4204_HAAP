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
    
    report_dict = classification_report(y_test, y_final_pred, output_dict=True)

    # Extract metrics for Class 0
    precision_0 = report_dict['0']['precision']
    recall_0 = report_dict['0']['recall']
    f1_0 = report_dict['0']['f1-score']
    support_0 = report_dict['0']['support']

    # Extract metrics for Class 1
    precision_1 = report_dict['1']['precision']
    recall_1 = report_dict['1']['recall']
    f1_1 = report_dict['1']['f1-score']
    support_1 = report_dict['1']['support']

    # Extract macro average metrics
    macro_avg_precision = report_dict['macro avg']['precision']
    macro_avg_recall = report_dict['macro avg']['recall']
    macro_avg_f1 = report_dict['macro avg']['f1-score']
    macro_avg_support = report_dict['macro avg']['support']

    # Extract weighted average metrics
    weighted_avg_precision = report_dict['weighted avg']['precision']
    weighted_avg_recall = report_dict['weighted avg']['recall']
    weighted_avg_f1 = report_dict['weighted avg']['f1-score']
    weighted_avg_support = report_dict['weighted avg']['support']

    # Construct the new report string
    new_report_output = f"""
Classification Performance Summary

The following report details the performance of the classification model, presenting key metrics for each class and overall model effectiveness.

| Metric     | Class 0 | Class 1 | Overall (Macro Avg) | Overall (Weighted Avg) |
|------------|---------|---------|---------------------|------------------------|
| Precision  | {precision_0:.2f}    | {precision_1:.2f}    | {macro_avg_precision:.2f}                | {weighted_avg_precision:.2f}                   |
| Recall     | {recall_0:.2f}    | {recall_1:.2f}    | {macro_avg_recall:.2f}                | {weighted_avg_recall:.2f}                   |
| F1-Score   | {f1_0:.2f}    | {f1_1:.2f}    | {macro_avg_f1:.2f}                | {weighted_avg_f1:.2f}                   |
| Support    | {support_0}      | {support_1}      | {macro_avg_support}                  | {weighted_avg_support}                     |

Overall Accuracy: {accuracy:.2f}

Explanation of Metrics:
*   Precision: The proportion of true positive predictions among all positive predictions for a given class. It indicates the model's ability to avoid false positives.
*   Recall: The proportion of true positive predictions among all actual positive instances for a given class. It indicates the model's ability to find all positive samples (avoid false negatives).
*   F1-Score: The harmonic mean of precision and recall, providing a single metric that balances both.
*   Support: The number of actual occurrences of each class in the specified dataset.
*   Accuracy: The proportion of correctly classified instances out of the total instances.
*   Macro Avg: The unweighted mean of the metric for each class, treating all classes equally.
*   Weighted Avg: The average of the metric for each class, weighted by their respective support (number of instances).
"""
    print(new_report_output)

    # Create visualization plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance
    if hasattr(model, 'estimators_'): # Handle ensemble models like VotingClassifier
        rf_model = None
        for name, estimator in model.estimators: # Changed from model.estimators_ to model.estimators
            if name == 'rf': # Assuming 'rf' is the name for RandomForestClassifier
                rf_model = estimator
                break
        
        if rf_model and hasattr(rf_model, 'feature_importances_'):
            plot_feature_importance(rf_model, feature_names, ax1)
        else:
            ax1.set_title('Random Forest Feature Importance Not Available')
            ax1.text(0.5, 0.5, 'Could not find Random Forest estimator or its feature_importances_ attribute in ensemble.', 
                     horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    elif hasattr(model, 'feature_importances_'): # Handle single tree-based models
        plot_feature_importance(model, feature_names, ax1)
    elif hasattr(model, 'coef_'): # Handle linear models like Logistic Regression
        # For linear models, coefficients can be used as a proxy for feature importance
        importance = np.abs(model.coef_[0])
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax1)
        ax1.set_title('Feature Importance (from coefficients)')
        ax1.set_xlabel('Absolute Coefficient Value')
        ax1.set_ylabel('Feature')
    else: # Feature importance not available for this model type
        ax1.set_title('Feature Importance Not Available')
        ax1.text(0.5, 0.5, 'Model does not have feature_importances_ or coef_ attribute.', 
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
