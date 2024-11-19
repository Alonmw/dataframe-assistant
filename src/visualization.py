"""
    Add some visualization for text features?
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .categorization import categorize_feature, categorize_all_features

matplotlib.use('TkAgg')


def plot_distribution(feature, ax=None):
    """
    Plots the distribution of a numerical feature.

    Parameters:
    - feature: The numerical feature (Pandas Series).
    - ax: Matplotlib Axes object (optional).

    Returns:
    - plots directly
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(feature, bins=30, kde=True, ax=ax, color='blue')
    ax.set_title(f"Distribution of {feature.name}")
    ax.set_xlabel(feature.name)
    ax.set_ylabel("Frequency")
    plt.show()

def plot_categorical_distribution(feature, ax=None):
    """
    Plots the frequency distribution of a categorical feature.

    Parameters:
    - feature: The categorical feature (Pandas Series).
    - ax: Matplotlib Axes object (optional).

    Returns:
    - plots directly
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=feature, ax=ax, palette='viridis')
    ax.set_title(f"Category Distribution of {feature.name}")
    ax.set_xlabel(feature.name)
    ax.set_ylabel("Count")
    plt.show()

def plot_correlation_matrix(df, ax=None):
    """
    Plots a heatmap of the correlation matrix for numerical features.

    Parameters:
    - df: The input DataFrame.
    - ax: Matplotlib Axes object (optional).

    Returns:
    - Plots directly
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    plt.show()

def summarize_dataframe(df):
    """
    Parameters:
    - df: The input DataFrame.

    Returns:
    - None: Displays plots directly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Data types distribution
    df.dtypes.value_counts().plot(kind='bar', ax=axes[0], color='teal')
    axes[0].set_title("Feature Data Types")
    axes[0].set_xlabel("Data Type")
    axes[0].set_ylabel("Count")
    # Missing data distribution
    missing = df.isnull().mean() * 100
    missing[missing > 0].sort_values().plot(kind='barh', ax=axes[1], color='orange')
    axes[1].set_title("Missing Data Percentage")
    axes[1].set_xlabel("Percentage")
    axes[1].set_ylabel("Features")
    plt.tight_layout()
    plt.show()

def plot_target_distribution(target):
    """
    HELPER FUNCTION
    Plots the distribution of the target variable.

    Parameters:
    - target: Target variable (Pandas Series).

    Returns:
    - None: Displays the plot directly.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(target, bins=30, kde=False, color='purple')
    plt.title("Target Distribution")
    plt.xlabel(target.name)
    plt.ylabel("Frequency")
    plt.show()

def _analyze_target_vs_numerical(feature, target):
    """
    HELPER FUNCTION
    Analyzes the relationship between a numerical feature and the target variable.

    Parameters:
    - feature: Numerical feature (Pandas Series).
    - target: Target variable (Pandas Series).

    Returns:
    - None: Displays plots and prints correlation.
    """
    correlation = feature.corr(target)
    print(f"Correlation between {feature.name} and {target.name}: {correlation:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot
    sns.scatterplot(x=feature, y=target, ax=axes[0], color='green')
    axes[0].set_title(f"{feature.name} vs {target.name}")
    axes[0].set_xlabel(feature.name)
    axes[0].set_ylabel(target.name)

    # Plot 2: Box plot of target grouped by feature quantiles
    quantiles = pd.qcut(feature, q=4, duplicates='drop')
    sns.boxplot(x=quantiles, y=target, ax=axes[1], palette='cool')
    axes[1].set_title(f"Box Plot: {target.name} by {feature.name} Quartiles")
    axes[1].set_xlabel(feature.name + " Quartiles")
    axes[1].set_ylabel(target.name)

    plt.tight_layout()
    plt.show()

def _analyze_target_vs_categorical(feature, target):
    """
    Analyzes the relationship between a categorical feature and the target variable.

    Parameters:
    - feature: Categorical feature (Pandas Series).
    - target: Target variable (Pandas Series).

    Returns:
    - None: Displays plots and summary statistics.
    """
    summary = target.groupby(feature).mean().sort_values()
    print("Average target value per category:")
    print(summary)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=summary.index, y=summary.values, palette='viridis')
    plt.title(f"Average {target.name} by {feature.name}")
    plt.xlabel(feature.name)
    plt.ylabel(f"Average {target.name}")
    plt.xticks(rotation=45)
    plt.show()

def analyze_target_vs_feature(feature, target):
    """
    Analyzes the relationship between any feature and the target variable.
    Parameters:
    :param feature: Pandas Series. A feature in the DataFrame.
    :param target: Pandas Series. A target variable in the DataFrame.

    :return: None: Displays plots and summary statistics.
    """
    feature_type = categorize_feature(feature)
    if feature_type == "Categorical":
        _analyze_target_vs_categorical(feature, target)
    elif feature_type == "Integer" or feature_type == "Float":
        _analyze_target_vs_numerical(feature, target)
    else:
        raise TypeError("Feature type must be either Categorical or Numerical.")


def summarize_target_relationships(df, target):
    """
    Summarizes the relationship of all features in the DataFrame with the target variable.

    Parameters:
    - df: DataFrame containing features.
    - target: Target variable (Pandas Series).

    Returns:
    - None: Displays key summaries and statistics.
    """
    print("Target Relationship Summary:")
    categories = ['Integer', 'Float', 'Categorical']
    for column in df.columns:
        print(f"\nAnalyzing {column}:")
        if categorize_feature(df[column]) in categories:
            analyze_target_vs_feature(df[column], target)

