"""
TODO:
    check_low_variance():
        *Needs to be adjusted to support other dtypes*
    Identify invalid data:
        Outliers for numerical features.
        Invalid or inconsistent values for categorical features.
    Provide a summary report:
        Summarize all the quality checks for easy interpretation.
"""

import pandas as pd
import numpy as np

def check_missing_data(df):
    """
    Identifies missing data in the DataFrame.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - missing_data: A DataFrame with columns, number of missing values, and percentage.
    """
    missing = df.isnull().sum()
    percentage = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': percentage
    }).sort_values(by='Percentage', ascending=False)
    return missing_data

def check_duplicates(df):
    """
    Identifies duplicate rows and columns.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - duplicate_info: A dictionary with counts of duplicate rows and columns.
    """
    duplicate_rows = df.duplicated().sum()
    duplicate_columns = df.T.duplicated().sum()
    return {
        'Duplicate Rows': duplicate_rows,
        'Duplicate Columns': duplicate_columns
    }

def check_low_variance(df, relative_threshold=0.01, absolute_threshold=5):
    """
    Identifies features with low variance.

    Parameters:
    - df: The input DataFrame.
    - threshold: Variance threshold to flag features.

    Returns:
    - low_variance_features: A list of features with variance below the threshold.
    """
    low_variance_features = []
    for col in df.columns:
        if df[col].nunique() / len(df[col]) <= relative_threshold or df[col].nunique() <= absolute_threshold:
            low_variance_features.append(col)
    return low_variance_features

def check_outliers(df, feature_name, method="IQR", threshold=3):
    """
    Detects outliers in a given feature using the specified method.

    Parameters:
    - df: The input DataFrame.
    - feature_name: The numerical feature to analyze.
    - method: The method to use for outlier detection ('IQR' or 'Z-score').

    Returns:
    - outliers: A DataFrame containing outlier rows.
    """
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the DataFrame.")

        # Ensure the feature is numerical
    if not np.issubdtype(df[feature_name].dtype, np.number):
        raise ValueError(f"Feature '{feature_name}' must be a numerical column.")
    feature = df[feature_name]
    if method == "IQR":
        Q1 = feature.quantile(0.25)
        Q3 = feature.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(feature < lower_bound) | (feature > upper_bound)]
    elif method == "Z-score":
        mean = feature.mean()
        std_dev = feature.std()
        z_scores = (feature - mean) / std_dev
        return df[(z_scores.abs() > threshold)]
    else:
        raise ValueError("Invalid method. Use 'IQR' or 'Z-score'.")

def generate_quality_report(df):
    """
    Generates a summary report of data quality.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - report: A dictionary summarizing data quality issues.
    """
    report = {
        'Missing Data': check_missing_data(df),
        'Duplicate Info': check_duplicates(df),
        'Low Variance Features': check_low_variance(df),
        # Add any additional checks here
    }
    return report
