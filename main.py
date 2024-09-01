# Imports
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

import mlflow

# Reading data
data = pd.read_csv("Titanic-Dataset.csv")
num_col: list[str] = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
cat_col: list[str] = data.select_dtypes(include=['object']).columns.to_list()


# Handling missing data
for col in cat_col:
    data[col].fillna(data[col].mode()[0], inplace=True)

for col in num_col:
    data[col].fillna(data[col].median(), inplace=True)

data = data.drop(['Name', 'Parch', 'Ticket', 'Cabin'], axis = 1)

# Converting categorical column to numerical columns using label encoding
for col in cat_col:
    le = preprocessing.LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop(['Survived', 'PassengerId'], 1)
y = data.Survived

SEED: int = 1

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

