import pandas as pd

from src.categorization import categorize_all_features
from src.text_analysis import generate_wordcloud, plot_text_length_distribution, analyze_text_vs_target, \
    summarize_text_analysis

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')

categorized = categorize_all_features(df)
text_series = df['neighbourhood_cleansed']
summarize_text_analysis(text_series, df['price'])