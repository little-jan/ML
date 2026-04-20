# practice code from d2l.ai
import torch
import pandas as pd
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

# reading the dataset
data = pd.read_csv(data_file)
print(data)

# NaN false and trues
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# fill in NaN values with average
inputs = inputs.fillna(inputs.mean())
print(inputs)

# conversion to tensor format
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(X, y)



# from UC Irvine Machine Learning Repository (for self practice)
from ucimlrepo import fetch_ucirepo

# fetch dataset
abalone = fetch_ucirepo(id=1)

# forces pandas display to show all columns without wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# data (as pandas dataframes)
X = abalone.data.features
y = abalone.data.targets

# metadata
print(abalone.metadata)

# variable information
print(abalone.variables)



# NaN into 0s and 1s
inpts, trgts = abalone.variables.iloc[:,0:2], abalone.variables.iloc[:,2]
inpts = pd.get_dummies(inpts, dummy_na=True)
print(inpts)

# calculate percentage of True/False values
missing_role_pct = inpts['role_nan'].mean() * 100
print(f"Percentage of missing values: {missing_role_pct}%")

# printing certain columns
selected_cols = abalone.variables[['name', 'type']]
print(selected_cols)