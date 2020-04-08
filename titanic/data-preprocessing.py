
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing dataset
dataset = pd.read_csv('titanic/datasets/full.csv')

# Separate dataset (X) and dependent vector (y)
X = dataset.drop(columns=['Survived'])
y = dataset.filter(items=['Survived'])


# Handle Missing values
## All Empty
X.dropna(axis=0,thresh=1, inplace=True)
X.reset_index(inplace=True)
X.drop(['index'], axis=1, inplace=True)

## Filling missing
X.fillna(X.mean(), inplace=True)
X['Embarked'].fillna(X['Embarked'].value_counts().index[0], inplace=True)
X['Sex'].fillna(X['Sex'].value_counts().index[0], inplace=True)


# Encode Categorical Data
dummy_vars = ['Sex', 'Embarked']
X_categorical = pd.get_dummies(X[dummy_vars])
X = pd.concat([X_categorical, X.drop(dummy_vars, axis=1)], axis = 1, sort=False)


# Splitting Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()

unscalable_cols = ['PassengerId','Name', 'Ticket', 'Cabin']

X_train_unscalable = X_train[unscalable_cols]
X_train_scalable = X_train.drop(unscalable_cols, axis=1)
X_train_scaled = ss_X.fit_transform(X_train_scalable)

X_train = pd.concat([X_train_unscalable, X_train_scalable], axis=1, sort=False)

X_test_unscalable = X_train[unscalable_cols]
X_test_scalable = X_test.drop(unscalable_cols, axis=1)
X_test_scaled = ss_X.transform(X_test_scalable)

X_test = pd.concat([X_test_unscalable, X_test_scalable], axis=1, sort=False)

