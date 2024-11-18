"""
    Analyze the distribution of the target variable.
    Examine how the target variable interacts with:
        Numerical features (e.g., correlation, average values per label).
        Categorical features (e.g., average target per category, distribution).
    Generate summary insights about the target variable’s behavior.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from src.categorization import categorize_all_features

matplotlib.use('TkAgg')

def plot_target_distribution(target):
    """
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

def analyze_target_vs_numerical(feature, target):
    """
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

def analyze_target_vs_categorical(feature, target):
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
    categories = categorize_all_features(df)
    for column in df.columns:
        print(f"\nAnalyzing {column}:")
        if column in categories['Float'] or column in categories['Integer']:
            analyze_target_vs_numerical(df[column], target)
        elif column in categories['Categorical']:
            analyze_target_vs_categorical(df[column], target)
