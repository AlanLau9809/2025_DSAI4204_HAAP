# association_rules.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from config import APRIORI_MIN_SUPPORT, APRIORI_MIN_CONFIDENCE, APRIORI_MIN_LIFT
import joblib
import matplotlib.pyplot as plt

def prepare_data_for_arm(df, categorical_cols):
    """
    Prepares the DataFrame for Association Rule Mining by one-hot encoding
    and converting to boolean format.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list): List of categorical column names to consider for ARM.
        
    Returns:
        pd.DataFrame: Transactional DataFrame with boolean values.
    """
    print("\n--- Preparing Data for Association Rule Mining ---")
    # Select only categorical columns for ARM
    df_arm = df[categorical_cols].copy()
    
    # One-hot encode the categorical columns
    df_encoded = pd.get_dummies(df_arm, columns=categorical_cols, prefix_sep='_is_')
    
    # Convert to boolean for Apriori
    df_encoded = df_encoded.astype(bool)
    
    print(f"Data prepared for ARM. Shape: {df_encoded.shape}")
    return df_encoded

def find_frequent_itemsets(df_encoded, min_support=None):
    """
    Finds frequent itemsets using the Apriori algorithm.
    
    Args:
        df_encoded (pd.DataFrame): Transactional DataFrame with boolean values.
        min_support (float, optional): The minimum support threshold. If None, uses APRIORI_MIN_SUPPORT.
        
    Returns:
        pd.DataFrame: DataFrame of frequent itemsets.
    """
    print("\n--- Finding Frequent Itemsets with Apriori ---")
    support = min_support if min_support is not None else APRIORI_MIN_SUPPORT
    
    frequent_itemsets = apriori(df_encoded, min_support=support, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={support}")
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=None, min_lift=None, save_path=None):
    """
    Generates association rules from frequent itemsets.
    
    Args:
        frequent_itemsets (pd.DataFrame): DataFrame of frequent itemsets.
        min_confidence (float, optional): The minimum confidence threshold. If None, uses APRIORI_MIN_CONFIDENCE.
        min_lift (float, optional): The minimum lift threshold. If None, uses APRIORI_MIN_LIFT.
        save_path (str, optional): Path to save the generated rules.
        
    Returns:
        pd.DataFrame: DataFrame of association rules.
    """
    print("\n--- Generating Association Rules ---")
    confidence = min_confidence if min_confidence is not None else APRIORI_MIN_CONFIDENCE
    lift = min_lift if min_lift is not None else APRIORI_MIN_LIFT
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
    rules = rules[rules['lift'] >= lift]
    
    print(f"Found {len(rules)} association rules with min_confidence={confidence} and min_lift={lift}")
    
    if save_path:
        joblib.dump(rules, save_path)
        print(f"Association rules saved to {save_path}")
        
    return rules

def analyze_rules(rules):
    """
    Prints and visualizes key metrics of association rules.
    
    Args:
        rules (pd.DataFrame): DataFrame of association rules.
    """
    print("\n--- Analyzing Association Rules ---")
    if rules.empty:
        print("No rules to analyze.")
        return
        
    print("\nTop 10 Rules by Lift:")
    print(rules.sort_values('lift', ascending=False).head(10))
    
    print("\nTop 10 Rules by Confidence:")
    print(rules.sort_values('confidence', ascending=False).head(10))
    
    # Basic visualization (can be expanded)
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence of Association Rules')
    plt.grid(True)
    plt.show()
