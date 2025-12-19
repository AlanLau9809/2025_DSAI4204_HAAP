# clustering.py
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import KMEANS_PARAMS, DBSCAN_PARAMS, RANDOM_STATE
import joblib

def train_kmeans_model(X, n_clusters=None, save_path=None):
    """
    Trains a K-Means clustering model.
    
    Args:
        X (pd.DataFrame): Input data for clustering.
        n_clusters (int, optional): The number of clusters to form. If None, uses KMEANS_PARAMS.
        save_path (str, optional): Path to save the trained model.
        
    Returns:
        KMeans: Trained KMeans model.
    """
    print("\n--- Training K-Means Model ---")
    params = KMEANS_PARAMS.copy()
    if n_clusters:
        params['n_clusters'] = n_clusters
    
    kmeans = KMeans(**params, n_init=10) # n_init to suppress warning
    kmeans.fit(X)
    
    if save_path:
        joblib.dump(kmeans, save_path)
        print(f"K-Means model saved to {save_path}")
        
    return kmeans

def train_dbscan_model(X, eps=None, min_samples=None, save_path=None):
    """
    Trains a DBSCAN clustering model.
    
    Args:
        X (pd.DataFrame): Input data for clustering.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. If None, uses DBSCAN_PARAMS.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. If None, uses DBSCAN_PARAMS.
        save_path (str, optional): Path to save the trained model.
        
    Returns:
        DBSCAN: Trained DBSCAN model.
    """
    print("\n--- Training DBSCAN Model ---")
    params = DBSCAN_PARAMS.copy()
    if eps:
        params['eps'] = eps
    if min_samples:
        params['min_samples'] = min_samples
        
    dbscan = DBSCAN(**params)
    dbscan.fit(X)
    
    if save_path:
        joblib.dump(dbscan, save_path)
        print(f"DBSCAN model saved to {save_path}")
        
    return dbscan

def find_optimal_kmeans_clusters(X, max_clusters=10):
    """
    Finds the optimal number of clusters for K-Means using the Elbow Method and Silhouette Score.
    
    Args:
        X (pd.DataFrame): Input data for clustering.
        max_clusters (int): Maximum number of clusters to test.
        
    Returns:
        tuple: (list of inertias, list of silhouette scores)
    """
    print("\n--- Finding Optimal K-Means Clusters ---")
    inertias = []
    silhouette_scores = []
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        if i > 1: # Silhouette score requires at least 2 clusters
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None) # Placeholder for k=1
            
    # Plot Elbow Method
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_clusters + 1))
    
    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores[1:], marker='o')
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_clusters + 1))
    plt.tight_layout()
    plt.show()
    
    return inertias, silhouette_scores

def evaluate_clustering(X, labels):
    """
    Evaluates clustering results using Silhouette Score and Davies-Bouldin Index.
    
    Args:
        X (pd.DataFrame): Input data.
        labels (array-like): Cluster labels for each sample.
        
    Returns:
        tuple: (silhouette_avg, davies_bouldin_avg)
    """
    print("\n--- Evaluating Clustering Results ---")
    if len(set(labels)) < 2 or len(set(labels)) > len(X) - 1:
        print("Cannot compute Silhouette Score or Davies-Bouldin Index for less than 2 or more than n-1 clusters.")
        return None, None
        
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin_avg = davies_bouldin_score(X, labels)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg:.4f}")
    
    return silhouette_avg, davies_bouldin_avg
