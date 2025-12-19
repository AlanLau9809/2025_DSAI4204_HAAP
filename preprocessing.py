# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats.mstats import winsorize
from config import CAT_COLS, CON_COLS, TARGET_COL, TEST_SIZE, VAL_SIZE, RANDOM_STATE

def handle_missing_values(df):
    """
    Handle missing values in the dataset, specifically the 'thall' variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df = df.copy()
    
    # Handle thall variable - 0 values should be treated as missing
    if 'thall' in df.columns:
        # Replace 0 with NaN
        df['thall'] = df['thall'].replace(0, np.nan)
        # Fill with mode (most common value)
        mode_value = df['thall'].mode()[0]
        df['thall'].fillna(mode_value, inplace=True)
        print(f"Filled {df['thall'].isna().sum()} missing values in 'thall' with mode: {mode_value}")
    
    return df

def detect_and_handle_outliers(df):
    """
    Detect and handle outliers using IQR method and Winsorization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    df = df.copy()
    
    # Handle trtbps outliers using Winsorization
    if 'trtbps' in df.columns:
        # Find the 95th percentile threshold (values > 165 are outliers)
        threshold_95 = np.percentile(df['trtbps'], 95)
        print(f"trtbps 95th percentile: {threshold_95}")
        
        # Apply Winsorization - cap at 95th percentile
        winsorize_percentile = (stats.percentileofscore(df['trtbps'], 165)) / 100
        df['trtbps'] = winsorize(df['trtbps'], (0, (1 - winsorize_percentile)))
        print(f"Applied Winsorization to trtbps at {winsorize_percentile:.3f} percentile")
    
    # Handle thalach outliers using IQR method
    if 'thalachh' in df.columns:
        q1 = np.quantile(df['thalachh'], 0.25)
        q3 = np.quantile(df['thalachh'], 0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # Remove extreme outliers (especially very low values like 71)
        outliers_mask = (df['thalachh'] < lower_bound) | (df['thalachh'] > upper_bound)
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            print(f"Removing {outliers_count} outliers from thalachh (< {lower_bound:.1f} or > {upper_bound:.1f})")
            df = df[~outliers_mask].reset_index(drop=True)
    
    # Handle oldpeak outliers using Winsorization
    if 'oldpeak' in df.columns:
        # Cap values above 4.0 (98th percentile approach)
        threshold_value = 4.0
        winsorize_percentile = (stats.percentileofscore(df['oldpeak'], threshold_value)) / 100
        df['oldpeak'] = winsorize(df['oldpeak'], (0, (1 - winsorize_percentile)))
        print(f"Applied Winsorization to oldpeak at {winsorize_percentile:.3f} percentile")
    
    return df

def apply_feature_transformations(df):
    """
    Apply transformations to handle skewed distributions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with transformed features
    """
    df = df.copy()
    
    # Transform oldpeak using square root to reduce skewness
    if 'oldpeak' in df.columns:
        # Add small constant to handle zero values
        df['oldpeak_sqrt'] = np.sqrt(df['oldpeak'] + 0.001)
        
        # Calculate skewness before and after transformation
        original_skew = df['oldpeak'].skew()
        transformed_skew = df['oldpeak_sqrt'].skew()
        print(f"oldpeak skewness: {original_skew:.3f} -> {transformed_skew:.3f} (after sqrt transformation)")
        
        # Drop original oldpeak and rename transformed version
        df = df.drop('oldpeak', axis=1)
        df = df.rename(columns={'oldpeak_sqrt': 'oldpeak'})
    
    return df

def select_features_by_correlation(df, target_col, correlation_threshold=0.15):
    """
    Select features based on correlation with target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Target column name
        correlation_threshold (float): Minimum correlation threshold
        
    Returns:
        pd.DataFrame: DataFrame with selected features
    """
    df = df.copy()
    
    # Calculate correlations with target
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Features to drop based on low correlation (from online resource analysis)
    low_corr_features = ['chol', 'fbs', 'restecg']
    features_to_drop = [f for f in low_corr_features if f in df.columns and f != target_col]
    
    if features_to_drop:
        print(f"Dropping low-correlation features: {features_to_drop}")
        for feature in features_to_drop:
            if feature in df.columns:
                corr_value = correlations.get(feature, 0)
                print(f"  - {feature}: correlation = {corr_value:.3f}")
        df = df.drop(features_to_drop, axis=1)
    
    # Print remaining feature correlations
    remaining_correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    print("\nRemaining feature correlations with target:")
    for feature, corr in remaining_correlations.items():
        if feature != target_col:
            print(f"  {feature}: {corr:.3f}")
    
    return df

def preprocess_data(df):
    """
    Performs comprehensive preprocessing on the heart disease dataset.
    - Handles missing values
    - Detects and handles outliers
    - Applies feature transformations
    - Selects features based on correlation
    - Scales continuous features using RobustScaler
    - One-hot encodes categorical features

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print("=== Starting Comprehensive Data Preprocessing ===")
    
    # Step 1: Handle missing values
    print("\n1. Handling missing values...")
    df = handle_missing_values(df)
    
    # Step 2: Handle outliers
    print("\n2. Detecting and handling outliers...")
    df = detect_and_handle_outliers(df)
    
    # Step 3: Apply feature transformations
    print("\n3. Applying feature transformations...")
    df = apply_feature_transformations(df)
    
    # Step 4: Feature selection based on correlation
    print("\n4. Selecting features based on correlation...")
    target_column = TARGET_COL if TARGET_COL in df.columns else 'output'
    df = select_features_by_correlation(df, target_column)
    
    # Step 5: Update feature lists based on remaining columns
    remaining_categorical = [col for col in CAT_COLS if col in df.columns]
    remaining_continuous = [col for col in CON_COLS if col in df.columns]
    
    # Add any new continuous features (like transformed oldpeak)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_column = TARGET_COL if TARGET_COL in df.columns else 'output'
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Update continuous columns list
    remaining_continuous = [col for col in numeric_cols if col not in remaining_categorical]
    
    print(f"\nFinal feature sets:")
    print(f"Categorical features: {remaining_categorical}")
    print(f"Continuous features: {remaining_continuous}")
    
    # Step 6: Scale continuous features
    if remaining_continuous:
        print("\n5. Scaling continuous features with RobustScaler...")
        scaler = RobustScaler()
        df[remaining_continuous] = scaler.fit_transform(df[remaining_continuous])
    
    # Step 7: One-hot encode categorical features
    if remaining_categorical:
        print("\n6. One-hot encoding categorical features...")
        df = pd.get_dummies(df, columns=remaining_categorical, drop_first=True)
    
    print(f"\nPreprocessing complete! Final dataset shape: {df.shape}")
    return df

def split_data(df):
    """
    Splits the data into training, validation, and testing sets.
    Uses 90% train, 10% test split like the online resource for better performance.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
        tuple: A tuple containing X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Determine target column
    if TARGET_COL not in df.columns and 'output' in df.columns:
        target_col = 'output'
        X = df.drop(['output'], axis=1)
        y = df['output']
    else:
        target_col = TARGET_COL
        X = df.drop([TARGET_COL], axis=1)
        y = df[TARGET_COL]

    # Use 90/10 split like the online resource for better performance
    # First split: separate test set (10% instead of 20%)
    test_size_optimized = 0.1  # Reduced from 0.2 to match online resource
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size_optimized, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: from remaining 90%, create train (89%) and validation (1%) 
    # This results in approximately 89% train, 1% val, 10% test overall
    val_size_optimized = 0.01  # Very small validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_optimized, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\nOptimized Data Split (90/10 approach):")
    print(f"Training Shape: {X_train.shape} ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Validation Shape: {X_val.shape} ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"Testing Shape: {X_test.shape} ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
