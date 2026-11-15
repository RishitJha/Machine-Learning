1. Overview

Completed practical implementation of multiple supervised and unsupervised Machine Learning algorithms using Python and Scikit-learn.

Worked with NumPy, Pandas, Matplotlib, Seaborn and Scikit-learn for dataset preparation, model training, prediction, evaluation and visualization.

2. Algorithms Covered

A. Supervised Learning

Algorithm

Key Concepts / Implementation

Linear Regression

Predicting continuous values using linear relationships. Implemented 5-fold cross-validation.

Ridge Regression

L2 regularization; controls model complexity using alpha.

Lasso Regression

L1 regularization; can reduce some feature coefficients to zero.

Logistic Regression

Binary classification using probabilities and a logistic function.

Naive Bayes

Probabilistic classification using Bayes' theorem. Implemented Gaussian Naive Bayes.

K-Nearest Neighbors (KNN)

Classification based on the nearest data points; requires feature scaling.

Decision Tree

Classification through decision rules using Gini impurity or entropy.

Random Forest

Ensemble learning using multiple decision trees and majority voting (Bagging).

AdaBoost

Boosting technique combining weak learners, giving greater attention to previously misclassified samples.

B. Unsupervised Learning

Algorithm

Key Concepts / Implementation

K-Means Clustering

Divides data into K clusters using centroids.

Hierarchical Clustering

Builds a hierarchy of clusters using Agglomerative Clustering and Ward linkage.

DBSCAN

Density-based clustering using epsilon radius and minimum samples; identifies noise and outliers.

3. Important ML Concepts Practiced

Dataset Preparation: Loading datasets, extracting features (X) and target labels (Y).

Train-Test Split: Dividing datasets into 80% training and 20% testing using train_test_split().

Feature Scaling: Applying StandardScaler() to normalize feature distributions.

Model Training: Using .fit() to train models.

Prediction: Using .predict() and .fit_predict().

Hyperparameter Tuning: Using GridSearchCV to find suitable alpha values for Ridge and Lasso.

Cross Validation: Using 5-fold cross-validation to evaluate model performance.

Visualization: Plotting confusion matrices, decision trees, feature importance, centroids, dendrograms and clustering results.

4. Evaluation Metrics

Regression:

R² Score: Measures how well the model explains variance in the target.

MAE: Mean Absolute Error.

MSE: Mean Squared Error.

RMSE: Root Mean Squared Error.

Classification:

Accuracy Score

Precision

Recall

F1-Score

Confusion Matrix

Classification Report

5. Datasets Used

Boston Housing: Regression models and house-price prediction.

Breast Cancer Wisconsin: Logistic Regression, Random Forest and AdaBoost classification.

Iris Dataset: Naive Bayes, KNN, Decision Tree and K-Means clustering.

Make Blobs: Hierarchical clustering and dendrogram visualization.

Make Moons: DBSCAN density-based clustering.

6. Practical Results

Random Forest achieved approximately 96.49% accuracy on the Breast Cancer dataset.

AdaBoost achieved 97.37% accuracy on the same dataset.

Implemented confusion matrices and classification reports to evaluate classification performance.

Successfully generated dendrograms and clustering visualizations.

7. Libraries and Tools

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

SciPy

8. Current Learning Status

Completed: Practical implementation of regression, classification, ensemble learning and clustering algorithms.
