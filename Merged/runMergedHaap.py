# runHaap.py

# ==========================================
# HEART ATTACK PREDICTION PIPELINE (MERGED)
# ==========================================
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from dataloader import load_data
from preprocessing import preprocess_data, split_data
from train import train_ensemble_model, get_best_individual_model
from evaluate import final_report, plot_correlation_matrix
from config import VAL_SIZE

# Styling for plots
plt.style.use('ggplot')
sns.set_palette("husl")

def main(use_checkpoints, compare_individual=False):
    """
    Main function to run the optimized heart attack prediction pipeline.
    
    Args:
        use_checkpoints (bool): Whether to use saved model checkpoints
        compare_individual (bool): Whether to compare ensemble vs individual models
    """
    
    print("="*80)
    print("HEART ATTACK PREDICTION PIPELINE")
    print("="*80)

    # ==========================================
    # 1. DATA LOADING & EXPLORATION
    # ==========================================
    print("\nSTEP 1: DATA LOADING & EXPLORATION")
    print("-" * 50)
    
    df = load_data()
    print(f"Original dataset shape: {df.shape}")

    # Show correlation matrix
    print("\nGenerating correlation matrix...")
    plot_correlation_matrix(df)

    # ==========================================
    # 2. COMPREHENSIVE PREPROCESSING
    # ==========================================
    print("\nSTEP 2: COMPREHENSIVE PREPROCESSING")
    print("-" * 50)
    
    preprocessed_df = preprocess_data(df.copy())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(preprocessed_df)
    
    print(f"\nFinal preprocessed dataset shape: {preprocessed_df.shape}")
    print(f"Features after preprocessing: {X_train.shape[1]}")

    # ==========================================
    # 3. OPTIMIZED MODEL TRAINING
    # ==========================================
    print("\nSTEP 3: OPTIMIZED MODEL TRAINING")
    print("-" * 50)
    
    # Train ensemble model with optimized individual models
    final_model = train_ensemble_model(X_train, y_train, X_val, y_val, use_checkpoints=use_checkpoints)
    
    # Optional: Compare with best individual model
    if compare_individual:
        print("\nCOMPARING ENSEMBLE VS INDIVIDUAL MODELS")
        print("-" * 50)
        best_individual, best_name, best_score = get_best_individual_model(
            X_train, y_train, X_val, y_val, use_checkpoints
        )
        
        # Quick validation comparison
        ensemble_val_pred = final_model.predict(X_val)
        ensemble_val_score = (ensemble_val_pred == y_val).mean()
        
        print(f"\nValidation Comparison:")
        print(f"   Best Individual ({best_name}): {best_score:.4f}")
        print(f"   Ensemble Model:                {ensemble_val_score:.4f}")
        
        improvement = ensemble_val_score - best_score
        if improvement > 0:
            print(f"   Ensemble Advantage: +{improvement:.4f}")
        else:
            print(f"   Individual Advantage: +{abs(improvement):.4f}")

    # ==========================================
    # 4. FINAL EVALUATION ON TEST SET
    # ==========================================
    print("\nSTEP 4: FINAL EVALUATION ON TEST SET")
    print("-" * 50)
    
    accuracy, roc_auc = final_report(final_model, X_test, y_test)

    # ==========================================
    # 5. PERFORMANCE SUMMARY
    # ==========================================
    print("\nSTEP 5: PERFORMANCE SUMMARY")
    print("-" * 50)
    
    print(f"\nFINAL PERFORMANCE SUMMARY:")
    print(f"   Test Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ROC-AUC Score:     {roc_auc:.4f}")
    
    print(f"\nPipeline completed successfully!")
    return final_model, accuracy, roc_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate optimized heart disease prediction model.")
    parser.add_argument('--no-checkpoints', action='store_false', dest='use_checkpoints',
                        help="Set to train models from scratch instead of using checkpoints.")
    parser.add_argument('--compare-individual', action='store_true',
                        help="Compare ensemble performance with best individual model.")
    args = parser.parse_args()
    main(args.use_checkpoints, args.compare_individual)
