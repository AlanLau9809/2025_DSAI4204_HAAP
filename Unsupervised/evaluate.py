# evaluate.py
import pandas as pd
from models.clustering import evaluate_clustering, find_optimal_kmeans_clusters
from models.association_rules import analyze_rules
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_unsupervised_models(X_processed, kmeans_model, dbscan_model, association_rules_df):
    """
    Evaluates clustering and association rule mining results.
    
    Args:
        X_processed (pd.DataFrame): Preprocessed data used for clustering.
        kmeans_model (KMeans): Trained KMeans model.
        dbscan_model (DBSCAN): Trained DBSCAN model.
        association_rules_df (pd.DataFrame): Generated association rules.
    """
    print("\n" + "="*50)
    print("STEP 4: EVALUATING UNSUPERVISED MODELS")
    print("="*50)

    # --- Evaluate Clustering ---
    print("\n--- Evaluating K-Means Clustering ---")
    if kmeans_model:
        kmeans_labels = kmeans_model.labels_
        if len(set(kmeans_labels)) > 1: # Ensure more than one cluster for evaluation
            evaluate_clustering(X_processed, kmeans_labels)
            
            # Visualize K-Means clusters (example for 2 principal components)
            # This would require dimensionality reduction first, e.g., PCA
            # For now, a simple scatter plot if X_processed has few dimensions
            if X_processed.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=X_processed.iloc[:, 0], y=X_processed.iloc[:, 1], hue=kmeans_labels, palette='viridis', legend='full')
                plt.title('K-Means Clusters (first two features)')
                plt.xlabel(X_processed.columns[0])
                plt.ylabel(X_processed.columns[1])
                plt.show()
            elif X_processed.shape[1] == 1:
                plt.figure(figsize=(8, 6))
                sns.histplot(x=X_processed.iloc[:, 0], hue=kmeans_labels, palette='viridis', multiple='stack')
                plt.title('K-Means Clusters (first feature)')
                plt.xlabel(X_processed.columns[0])
                plt.show()
        else:
            print("K-Means resulted in only one cluster, skipping evaluation metrics.")
            
        # Find optimal K for K-Means
        find_optimal_kmeans_clusters(X_processed)

    print("\n--- Evaluating DBSCAN Clustering ---")
    if dbscan_model:
        dbscan_labels = dbscan_model.labels_
        n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) # Exclude noise points
        if n_clusters_ > 1: # Ensure more than one cluster for evaluation
            evaluate_clustering(X_processed, dbscan_labels)
            
            # Visualize DBSCAN clusters
            if X_processed.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=X_processed.iloc[:, 0], y=X_processed.iloc[:, 1], hue=dbscan_labels, palette='viridis', legend='full')
                plt.title('DBSCAN Clusters (first two features)')
                plt.xlabel(X_processed.columns[0])
                plt.ylabel(X_processed.columns[1])
                plt.show()
            elif X_processed.shape[1] == 1:
                plt.figure(figsize=(8, 6))
                sns.histplot(x=X_processed.iloc[:, 0], hue=dbscan_labels, palette='viridis', multiple='stack')
                plt.title('DBSCAN Clusters (first feature)')
                plt.xlabel(X_processed.columns[0])
                plt.show()
        else:
            print("DBSCAN resulted in zero or one cluster (or only noise points), skipping evaluation metrics.")

    # --- Evaluate Association Rules ---
    print("\n--- Analyzing Association Rules ---")
    if association_rules_df is not None and not association_rules_df.empty:
        analyze_rules(association_rules_df)
    else:
        print("No association rules to analyze.")
