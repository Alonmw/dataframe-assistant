import pandas as pd
from src.visualization import *
from src.data_quality import *

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')

analyze_target_vs_feature(df['beds'], df['price'])
