# dataloader.py
import pandas as pd
import numpy as np
from config import DATA_PATH, CLUSTER_LABELS_PATH

def load_data():
    """
    Loads the heart disease dataset, cleans it, handles duplicates, and returns a DataFrame.

    Returns:
        pd.DataFrame: The loaded and cleaned DataFrame.
    """
    df = pd.read_csv(DATA_PATH)

    # Load cluster labels and merge with the main DataFrame
    try:
        cluster_labels_df = pd.read_csv(CLUSTER_LABELS_PATH, index_col=0)
        df = df.merge(cluster_labels_df, left_index=True, right_index=True, how='left')
        print(f"Cluster labels loaded from {CLUSTER_LABELS_PATH} and merged.")
    except FileNotFoundError:
        print(f"Warning: Cluster labels file not found at {CLUSTER_LABELS_PATH}. Proceeding without cluster labels.")
    
    # The 'thall' column has a value of 0 which is not described in the data dictionary.
    # The article suggests this is a null value. We will replace it with NaN and then fill
    # it with the mode (which is 2), as done in the reference analysis.
    df['thall'] = df['thall'].replace(0, np.nan)
    df['thall'].fillna(df['thall'].mode()[0], inplace=True) # Using mode() is more robust
    
    print("--- Dataset Info ---")
    df.info()

    # CHECK FOR DUPLICATES
    dup_count = df.duplicated().sum()
    print(f"\nNumber of duplicates found: {dup_count}")
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print("Duplicates removed.")
        
    return df
