import pandas as pd

from src.data_quality import check_outliers
from test_categorization import *

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')
categorize = categorize_all_features(df)
outliers = check_outliers(df, 'price', method='Z-score')
df = df.drop(outliers.index)
summarize_target_relationships(df, df['price'])