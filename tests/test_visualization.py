import pandas as pd
from src.visualization import *
from src.data_quality import *

df = pd.read_csv(r'C:\Users\User\PycharmProjects\dataframe-assistant\data\listings.csv')

plot_correlation_matrix(df)
