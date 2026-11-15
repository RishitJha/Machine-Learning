#IMPORTING ALGORITHM

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#1. DATASET LOADING:
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target
print(X)
print()
print()
print(Y)
print()
print()
#2. Split into training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)

#3. TRAINING:
# Base learner define(Decision stump: depth=1)
# Adaboost by default uses DecisionTreeClassifier(max_depth=1) u
base_estimator = DecisionTreeClassifier(max_depth=1)
ada_model = AdaBoostClassifier(estimator= base_estimator,
                               n_estimators=100,
                               learning_rate =1.0,
                               random_state = 42)

# 4. Fitting the model
ada_model.fit(X_train, Y_train)

# PREDICTION:
Y_pred = ada_model.predict(X_test)
print(Y_pred)
print()
print()
print(Y_test)

#ACCURACY:
print(f"AdaBoost Accuracy Score: {accuracy_score(Y_test, Y_pred)*100:.2f}%")
print()
print()
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print()
print()
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print()
print()
