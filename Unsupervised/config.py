# config.py

# File path for the dataset
DATA_PATH = "/Users/chunmanchan/Downloads/Alan/DSAI4204/archive/heart.csv"

# List of categorical and continuous feature names
CAT_COLS = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
CON_COLS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

# Random state for reproducibility
RANDOM_STATE = 42

# Clustering Hyperparameters
KMEANS_PARAMS = {
    'n_clusters': 3, # Example: number of clusters
    'random_state': RANDOM_STATE
}
DBSCAN_PARAMS = {
    'eps': 0.5,      # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    'min_samples': 5 # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
}

# Association Rule Mining (Apriori) Parameters
APRIORI_MIN_SUPPORT = 0.05
APRIORI_MIN_CONFIDENCE = 0.7
APRIORI_MIN_LIFT = 1.2

# Checkpoint paths
KMEANS_CHECKPOINT = "Unsupervised/checkpoints/kmeans_model.joblib"
DBSCAN_CHECKPOINT = "Unsupervised/checkpoints/dbscan_model.joblib"
APRIORI_CHECKPOINT = "Unsupervised/checkpoints/apriori_rules.pkl"
CLUSTER_LABELS_PATH = "Unsupervised/checkpoints/cluster_labels.csv"
