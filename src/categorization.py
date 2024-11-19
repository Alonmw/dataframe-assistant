"""
    TODO:
        Determine whether categorical that has 2 unique values is Boolean
        e.g - feature with values "yes" or "no"
"""
"""
    It is recommended to check categorization manually and make adjustments accordingly
    
    The check low variance wont be as useful when working with small datasets
    where number of examples is less than a 1000
"""
import pandas as pd
import numpy as np
from src.data_quality import feature_low_variance


def categorize_feature(feature):
    """
    Classify the feature by its datatype.

    Parameters:
    - df: DataFrame containing the feature
    - feature_name: name of the feature to categorize

    Returns:
    - category: A string representing the feature's category
    """

    # Check for low variance potentially categorical
    if feature_low_variance(feature):
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
    for col in df:
        classification = categorize_feature(df[col])
        categories[classification].append(col)
    return categories
