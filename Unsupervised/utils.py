import pandas as pd

def save_cluster_labels(df_with_labels, save_path):
    """
    Saves the DataFrame with cluster labels to a CSV file.
    The DataFrame should contain the original index and a 'cluster_label' column.
    """
    df_with_labels.to_csv(save_path, index=True)
    print(f"Cluster labels saved to {save_path}")
