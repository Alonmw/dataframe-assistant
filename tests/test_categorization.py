import pandas as pd

from src.categorization import *

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')
print(categorize_all_features(df))
