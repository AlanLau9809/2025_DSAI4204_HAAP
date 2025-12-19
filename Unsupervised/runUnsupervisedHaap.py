# runUnsupervisedHaap.py

# ==========================================
# HEART ATTACK UNSUPERVISED LEARNING PIPELINE
# ==========================================
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd # Added pandas import
warnings.filterwarnings('ignore')

from dataloader import load_data
from preprocessing import preprocess_data
from train import train_unsupervised_models
from evaluate import evaluate_unsupervised_models
from utils import save_cluster_labels
from config import CLUSTER_LABELS_PATH

# Styling for plots
plt.style.use('ggplot')
sns.set_palette("husl")

def main():
    """
    Main function to run the unsupervised heart attack analysis pipeline.
    """
    
    print("="*80)
    print("HEART ATTACK UNSUPERVISED LEARNING PIPELINE")
    print("="*80)

    # ==========================================
    # 1. DATA LOADING & EXPLORATION
    # ==========================================
    print("\nSTEP 1: DATA LOADING & EXPLORATION")
    print("-" * 50)
    
    df = load_data()
    print(f"Original dataset shape: {df.shape}")

    # ==========================================
    # 2. COMPREHENSIVE PREPROCESSING
    # ==========================================
    print("\nSTEP 2: COMPREHENSIVE PREPROCESSING")
    print("-" * 50)
    
    preprocessed_df = preprocess_data(df.copy())
    
    print(f"\nFinal preprocessed dataset shape: {preprocessed_df.shape}")
    print(f"Features after preprocessing: {preprocessed_df.shape[1]}")

    # ==========================================
    # 3. UNSUPERVISED MODEL TRAINING
    # ==========================================
    kmeans_model, dbscan_model, association_rules_df = train_unsupervised_models(preprocessed_df, df.copy())

    # Save K-Means cluster labels
    if kmeans_model:
        # Get cluster labels from the trained K-Means model
        cluster_labels = kmeans_model.labels_
        
        # Create a DataFrame with original index and cluster labels
        df_cluster_labels = pd.DataFrame({'cluster_label': cluster_labels}, index=preprocessed_df.index)
        
        # Save the cluster labels to a file
        save_cluster_labels(df_cluster_labels, CLUSTER_LABELS_PATH)

    # ==========================================
    # 4. EVALUATION OF UNSUPERVISED MODELS
    # ==========================================
    evaluate_unsupervised_models(preprocessed_df, kmeans_model, dbscan_model, association_rules_df)

    print(f"\nPipeline completed successfully!")

if __name__ == '__main__':
    main()
