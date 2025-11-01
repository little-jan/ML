import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
y = df['logS']
x = df.drop('logS', axis = 1)
print(x)