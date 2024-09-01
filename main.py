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

#dropping some columns
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

#____________________Logisitic Regression________________________

lr = LogisticRegression(random_state=SEED)

lr_param_grid = {
    'C': [100, 10, 1, 0.1, 0.01],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr_gs = GridSearchCV(
    estimator=lr,
    param_grid=lr_param_grid,
    cv = 5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1,
)

lr_model = lr_gs.fit(X_train, y_train)

#____________________Decision Tree________________________

dt = DecisionTreeClassifier(
    random_state=SEED
)

dt_param_grid = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    "criterion": ["gini", "entropy"],
}

dt_gs = GridSearchCV(
    estimator=dt,
    param_grid=dt_param_grid,
    cv = 5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=0
)

dt_model = dt_gs.fit(X_train, y_train)

#____________________Random Forest_________________________

rf = RandomForestClassifier(random_state=SEED)

rf_param_grid = {
    'n_estimators':[400, 700],
    'max_depth': [15, 20, 20],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [50, 100]
}

rf_gs = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

rf_model = rf_gs.fit(X_train, y_train)

