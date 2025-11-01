import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
y = df['logS']
x = df.drop('logS', axis = 1)

# training set will have 80% of the data, testing set will have 20% of the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

lr = LinearRegression()
lr.fit(x_train, y_train)

# need to compare y_lr_train to y_train
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)