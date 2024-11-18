"""
TODO:
    Create visualizations for numerical features:
        Distribution plots.
        Correlation with the target.
    Visualize categorical features:
        Bar charts for frequency distribution.
    Analyze relationships between features:
        Scatter plots for numerical features.
        Heatmaps for correlation matrices.
    Provide summary visualizations for the DataFrame:
        Overall data distribution.
        Feature-wise data types.

    Add some visualization for text features?
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
