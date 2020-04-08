
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('datasets/data-preprocessing-data.csv')

# Separate dataset (X) and dependent vector (y)
X = dataset.drop(columns=['Purchased'])
y = dataset.filter(items=['Purchased'])

# Handle Missing values
## All Empty
X.dropna(axis=0,thresh=1, inplace=True)
X.reset_index(inplace=True)
X.drop(['index'], axis=1, inplace=True)

## Filling missing
X.fillna(X.mean(), inplace=True)
X['Country'].fillna(X['Country'].value_counts().index[0], inplace=True)

# Encode Categorical Data
dummy_vars = ['Country']
X_categorical = pd.get_dummies(X[dummy_vars])
X = pd.concat([X_categorical, X.drop(columns=dummy_vars)], axis = 1, sort=False)

y.replace({'Yes': 1, 'No': 0}, inplace=True)

# Splitting Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()

X_train= ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)