"""
    Analyze text data for basic statistics (e.g., word count, unique words).
    Visualize text distributions (e.g., word clouds, text length distributions).
    Examine relationships between textual features and the target variable.
    Provide summary insights on the text feature.
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
import seaborn as sns
matplotlib.use('TkAgg')


def text_statistics(text_series):
    """
    Compute basic statistics for a text column.

    Parameters:
    - text_series: Pandas Series containing text data.

    Returns:
    - stats: Dictionary with text statistics.
    """
    word_counts = text_series.str.split().apply(len)
    unique_word_counts = text_series.str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else 0)

    stats = {
        "Total entries": len(text_series),
        "Empty entries": text_series.isna().sum(),
        "Average word count": word_counts.mean(),
        "Median word count": word_counts.median(),
        "Average unique words": unique_word_counts.mean()
    }
    return stats


def plot_text_length_distribution(text_series):
    """
    Plot the distribution of text lengths.

    Parameters:
    - text_series: Pandas Series containing text data.

    Returns:
    - None: Displays the plot.
    """
    text_lengths = text_series.str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(text_lengths, bins=30, kde=True, color='skyblue')
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.show()


def generate_wordcloud(text_series):
    """
    Generate a word cloud from text data.

    Parameters:
    - text_series: Pandas Series containing text data.

    Returns:
    - None: Displays the word cloud.
    """
    all_text = " ".join(text_series.dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()

def summarize_text_analysis(text_series, target=None):
    """
    Summarize the text feature with or without relation to the target variable.

    Parameters:
    - text_series: Pandas Series containing text data.
    - target: Optional. Target variable (Pandas Series).

    Returns:
    - None: Prints summary and displays plots.
    """
    print("Text Statistics:")
    stats = text_statistics(text_series)
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Visualizations
    print("\nVisualizations:")
    plot_text_length_distribution(text_series)
    generate_wordcloud(text_series)

