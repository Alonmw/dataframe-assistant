from src.categorization import categorize_all_features
import pandas as pd

from src.visualization import plot_categorical_distribution

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')
categorized = categorize_all_features(df)
for feature in categorized['Categorical']:
    plot_categorical_distribution(df[feature])