import pandas as pd

from src.data_quality import *

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')
"""
print(check_missing_data(df))
print(check_duplicates(df))
print(check_low_variance(df))
print(check_outliers(df, 'price', 'Z-score'))
"""
print(generate_quality_report(df))
