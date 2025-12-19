# train.py
import pandas as pd
from models.clustering import train_kmeans_model, train_dbscan_model
from models.association_rules import find_frequent_itemsets, generate_association_rules
from config import KMEANS_CHECKPOINT, DBSCAN_CHECKPOINT, APRIORI_CHECKPOINT, CAT_COLS

def train_unsupervised_models(X_processed, df_original_for_arm):
    """
    Trains clustering and association rule mining models.
    
    Args:
        X_processed (pd.DataFrame): Preprocessed data for clustering.
        df_original_for_arm (pd.DataFrame): Original DataFrame (or relevant columns) for ARM.
        
    Returns:
        tuple: Trained KMeans model, trained DBSCAN model, generated association rules.
    """
    print("\n" + "="*50)
    print("STEP 3: TRAINING UNSUPERVISED MODELS")
    print("="*50)

    # --- Clustering Models ---
    print("\n--- Training Clustering Models ---")
    
    # K-Means
    kmeans_model = train_kmeans_model(X_processed, save_path=KMEANS_CHECKPOINT)
    
    # DBSCAN
    dbscan_model = train_dbscan_model(X_processed, save_path=DBSCAN_CHECKPOINT)
    
    # --- Association Rule Mining ---
    print("\n--- Training Association Rule Mining Model ---")
    
    # Prepare data for ARM (using original categorical columns)
    df_arm_prepared = df_original_for_arm[CAT_COLS].copy()
    # For ARM, we need to ensure all categorical columns are treated as items.
    # If there are continuous columns, they would need to be binned first.
    # For simplicity, we'll use the original categorical columns as defined in config.
    
    # One-hot encode categorical features for ARM
    df_arm_encoded = pd.get_dummies(df_arm_prepared, columns=CAT_COLS, prefix_sep='_is_')
    df_arm_encoded = df_arm_encoded.astype(bool)

    frequent_itemsets = find_frequent_itemsets(df_arm_encoded)
    association_rules_df = generate_association_rules(frequent_itemsets, save_path=APRIORI_CHECKPOINT)
    
    return kmeans_model, dbscan_model, association_rules_df
