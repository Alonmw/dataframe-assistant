import pandas as pd
import numpy as np
from src.data_quality import check_low_variance


def categorize_feature(df, feature_name):
    """
    Classify the feature by its datatype.

    Parameters:
    - df: DataFrame containing the feature
    - feature_name: name of the feature to categorize

    Returns:
    - category: A string representing the feature's category
    """
    feature = df[feature_name]

    # Check for low variance potentially categorical
    if feature_name in check_low_variance(df, relative_threshold=0.005):
        return 'Categorical'
    # Check for numerical data types
    if pd.api.types.is_numeric_dtype(feature):
        if np.any(np.floor(feature) == feature):  # If the values are integers
            return 'Integer'
        else:  # If the values are floats
            return 'Float'
    # Check for boolean data types
    elif pd.api.types.is_bool_dtype(feature):
        return 'Boolean'

    # Check for datetime data types
    elif pd.api.types.is_datetime64_any_dtype(feature):
        return 'Datetime'

    # Check for categorical data types
    elif isinstance(feature.dtype, pd.CategoricalDtype):
        return 'Categorical'

    # If the feature is a string object, check if it is text-based
    elif feature.dtype == 'object':
        return 'Text'

    # Handle complex or mixed data types
    elif isinstance(feature.iloc[0], (list, dict)):
        return 'Mixed'

    # Default case: if the type is unknown
    return 'Unknown'


def categorize_all_features(df):
    """
    Classify all features in the DataFrame by datatype.

    Parameters:
    - df: DataFrame containing the features to categorize

    Returns:
    - categories: A dictionary of feature names and their corresponding categories
    """
    categories = {
        'Integer': [],
        'Float': [],
        'Boolean': [],
        'Datetime': [],
        'Categorical': [],
        'Mixed': [],
        'Text': [],
        'Unknown': []
    }
    for feature_name in df.columns:
        classification = categorize_feature(df, feature_name)
        categories[classification].append(feature_name)
    return categories
