'''
==> task 1 
- input 
penguins.csv

- Task description 
Preprocess the penguin dataset by handling missing values and encoding categorical variables.

- output
penguins_preprocessed.csv       3 files 
every 2 classes in the dataset will be saved in a separate file.
'''
import pandas as pd

df = pd.read_csv('../Data/penguins.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.shape)