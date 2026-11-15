# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.datasets import fetch_openml
# boston = fetch_openml(name = 'boston', version = 1, as_frame= True, parser = 'liac-arff')
# # print(boston)

# df = boston.frame
# print(df.head())
# print()
# df['price'] = boston.target
# # print(df.head())
# # #Dividing the dataset into dependent and independent features(x, y)
# X = df.iloc[ :,:-1]
# # print(X.head())

# # [5 rows x 15 columns]
# #       CRIM    ZN  INDUS CHAS  ...  PTRATIO       B  LSTAT  MEDV
# # 0  0.00632  18.0   2.31    0  ...     15.3  396.90   4.98  24.0
# # 1  0.02731   0.0   7.07    0  ...     17.8  396.90   9.14  21.6
# # 2  0.02729   0.0   7.07    0  ...     17.8  392.83   4.03  34.7
# # 3  0.03237   0.0   2.18    0  ...     18.7  394.63   2.94  33.4
# # 4  0.06905   0.0   2.18    0  ...     18.7  396.90   5.33  36.2

# # [5 rows x 14 columns]
# # <bound method NDFrame.head of 0      24.0
# Y = df.iloc[:,-1]
# # print(Y.head)


# #OUTPUT:
# # 1      21.6
# # 2      34.7
# # 3      33.4
# # 4      36.2
# #        ... 
# # 501    22.4
# # 502    20.6
# # 503    23.9
# # 504    22.0
# # 505    11.9
# # Name: price, Length: 506, dtype: float64>

# #LINEAR REGRESSION
# from sklearn.linear_model import LinearRegression #importing linear regression functions which was mathematically explained in the video)
# from sklearn.model_selection import cross_val_score#importin cross validation function which allows for the cross validation of the training data to the predicted data
# X = X.astype(float) #converting all the independent data to float values
# Y = Y.astype(float) #coverting all the dependent data to float data
# lin_reg = LinearRegression()
# mse = cross_val_score(lin_reg, X, Y, scoring = 'neg_mean_squared_error', cv = 5) #scoring the error values as square to aavoid neg values, and the 5 fold cross validation causes data was divided into 5 parts, 4 parts for training, 1 part for testing)
# print(mse)

# #Since we need the mean square error mean value, we use
# mean_mse = np.mean(mse)
# print("MSE scores per fold:", mse)
# print("Average Negative MSE", mean_mse)


# #Implementing cross validation

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42) #the split divides the data step size as 42 spaces allowing enough space for the data to run, work and not be mixed)
# lin_reg.fit(X_train, Y_train)
# prediction = lin_reg.predict(X_test)
# # print(prediction)

# comparison_df = pd.DataFrame({'Actual': Y_test, 'Predicted': prediction})
# print(comparison_df.head())

# # Output:

# #Actual/VS predicted
# #      Actual  Predicted
# # 173    23.6       23.6
# # 274    32.4       32.4
# # 491    13.6       13.6
# # 72     22.8       22.8
# # 452    16.1       16.1

# from sklearn import metrics

# #1 R sqaured socre(best possible score is 1.0)
# r2 = metrics.r2_score(Y_test, prediction)

# #2 Mean abs error(avg error in price units)
# mae = metrics.mean_absolute_error(Y_test, prediction)

# #3 root mean squared error(standrad deviation of prediction error)
# rmse = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
# print(f"R-squared ScoreL: {r2:.4f}")
# print(f"Mean Absolute Error(MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# # #Output:
# #       CRIM    ZN  INDUS CHAS  ...  PTRATIO       B  LSTAT  MEDV
# # 0  0.00632  18.0   2.31    0  ...     15.3  396.90   4.98  24.0
# # 1  0.02731   0.0   7.07    0  ...     17.8  396.90   9.14  21.6
# # 2  0.02729   0.0   7.07    0  ...     17.8  392.83   4.03  34.7
# # 3  0.03237   0.0   2.18    0  ...     18.7  394.63   2.94  33.4
# # 4  0.06905   0.0   2.18    0  ...     18.7  396.90   5.33  36.2

# # [5 rows x 14 columns]

# # [-3.56224836e-28 -6.52614473e-28 -6.17342237e-29 -7.40732579e-28
# #  -4.03966297e-27]
# # MSE scores per fold: [-3.56224836e-28 -6.52614473e-28 -6.17342237e-29 -7.40732579e-28
# #  -4.03966297e-27]
# # Average Negative MSE -1.1701938164506492e-27
# #      Actual  Predicted
# # 173    23.6       23.6
# # 274    32.4       32.4
# # 491    13.6       13.6
# # 72     22.8       22.8
# # 452    16.1       16.1
# # R-squared ScoreL: 1.0000
# # Mean Absolute Error(MAE): 0.0000
# # Root Mean Squared Error (RMSE): 0.0000

# #RIDGE REGRESSION
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# ridge = Ridge()
# params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}
# ridge_regressor = GridSearchCV(ridge, params, scoring = 'neg_mean_squared_error', cv=5)
# ridge_regressor.fit(X,Y)
# print(ridge_regressor.best_params_)
# print(ridge_regressor.best_score_)

# #Output:
# # {'alpha': 1e-15}
# # -2.367301282032148e-28

# #LASSO REGRESSION:
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
# lasso = Lasso()
# params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}
# lasso_regressor = GridSearchCV(lasso, params, scoring = 'neg_mean_squared_error', cv=5)
# lasso_regressor.fit(X,Y)
# print(lasso_regressor.best_params_)
# print(lasso_regressor.best_score_)

# #Output:
# # {'alpha': 1e-08}
# # -1.0626085889427681e-07

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score

# #ENSURE DATA IS NUMERIC
# X = X.astype(float)
# Y = Y.astype(float)

# #1. scale the features(essential for ridge and lasso)
# scaler = StandardScaler() #scales weight as per ridge and lasso and may give more priority to some in case for ridge and may remove some weights in case of lasso)
# X_scaled = scaler.fit_transform(X)

# #2. Split into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 42)

# from sklearn.linear_model import LinearRegression
# #Initialise and train
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, Y_train)

# #Predict and calculate accuracy
# lin_pred = lin_reg.predict(X_test)
# lin_acc = r2_score(Y_test, lin_pred)*100
# print(f"Linear Regression Accuracy: {lin_acc:.2f}%")


# from sklearn.linear_model import Ridge
# #Initialise and train
# #alpha is the penalty strength(higher alpha= more restriction)
# ridge_reg = Ridge(alpha=10.0)
# ridge_reg.fit(X_train, Y_train)

# #Predict and calculate accuracy
# ridge_pred = ridge_reg.predict(X_test)
# ridge_acc = r2_score(Y_test, ridge_pred)*100
# print(f"Ridge Regression Accuracy: {ridge_acc:.2f}%")


# #LASSO REGRESSION
# from sklearn.linear_model import Lasso
# #Initialise and train
# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(X_train, Y_train)

# #Predict and calculate accuracy
# lasso_pred = lasso_reg.predict(X_test)
# lasso_acc = r2_score(Y_test, lasso_pred)*100
# print(f"Ridge Regression Accuracy: {ridge_acc:.2f}%")

#Output:Linear Regression Accuracy: 100.00%
# Ridge Regression Accuracy: 99.76%
# Ridge Regression Accuracy: 99.76%


######LOGISTIC REGRESSION(Only deals in between 0 and 1)


# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# #LogisticRegression: Main algortithm which does classification(Malignant ya Benign)
# from sklearn.preprocessing import StandardScaler
# ##StandardScaler: Numbers are brought to a scale between 0 and 1
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# #Metrics: checks how well the model works

# #Load Dataset
# data = load_breast_cancer()
# print(data)
# X = pd.DataFrame(data.data, columns = data.feature_names)
# #(features): Tumour properties(radius, texture, etc...)
# print(X)
# Y = data.target
# #Y targetL answer keys, 0 means it has cancer(malignant) and 1 means it doesn;t have cancer(Benign)
# print(Y)

# #Training and Testing data split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# #Test_size = 0.2, 2o percent data to test and 80 percen data to train
# #Random state= 42 one seed number, when we run the code, the data splits in the same way


# #FEATURE SCALING(NUMBERS DIVIDING EQUALLY)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# #Problem: some features are large(eg area= 1---) and some are small)eg: smoothness=0.1)

# #Soltuion: standard scaler bring every scale into a standard range so that model foesn;t get confued
# #fit_transform: learns from training data and chages itself
# #transform: transforms test data similarly


# #MODEL TRAIN:
# model =LogisticRegression()
# model.fit(X_train, Y_train)

# #LogisticRgeression(): One empty model is made
# #model.fit: Learning or padhai
# #Model X_train(question) and Y_train(answer) learns from patterns

# # Predictions
# Y_pred =model.predict(X_test)
# print(Y_pred)
# print()
# print()
# print(Y_test)

# # Accuracy:
# print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred):.4f}")
# print(classification_report(Y_test, Y_pred))

# #accuracy_score: tells how many times the answer is correct, in every 100 tests
# #classification report: tells in details
# #Precision: How many people were mistakenly identified as ill
# #Recall: How many really ill people were identified

# #CONFUSION MATRIX
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# #1. Confusion matrix calculate karna
# # Y_test = Asli jawab, Y_test = Model ke guesses
# cm = confusion_matrix(Y_test, Y_pred)

# #2 Visual Representation create karna
# # Display_labels mein hum "Malignant" aur "Benign" dikhayenge
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)

# #3. Plot draw karna
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix:Model Performance")
# plt.show()


#NAIVE BAYES ALGORITHM

#Ml model by which Iris flowers are studied and then there species is to be predicted as per it
#Gaussian Naive Bayes Algorithm is used and it works on probability of dependent and independet events

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

#sklearnL ML mainlibrary , dataset, dat split and naive bayes algorithm
#pandas: data is to be visualisez in the format or dataframe
#seaborn/latplotlib: Result is visualized in the form of confusion matrix


#DATA SETLOAD
# iris = datasets.load_iris()
# # print(iris)

# # 0 = setosa
# # 1 = versicolor
# # 2 = virginica

# X = iris.data #Features: Lnegth and width
# Y = iris.target #Labels: Phool ko naam(0,1,2)

# #Iris dataset loaded, X inputted with features
# #petal length, Y has answers(which flower is it)
# print(X)
# print()
# print(Y)
# print()

# #Convert to dataframe just for a cleaner look at the data
# df = pd.DataFrame(X, columns=iris.feature_names)
# df['species'] = Y
# print("First 6 rows of the dataset:")
# print(df.head())

# #Train Test Split:
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# #Model Training
# model = GaussianNB() #Naive Bayes is called as an object
# model.fit(X_train, Y_train) #.fit real training happens here, X_train and Y_train learns the pattern)

# #Prediction:
# Y_pred = model.predict(X_test)
# print(Y_pred)
# print()
# print(Y_test)

# # Accuracy:
# accuracy = accuracy_score(Y_test, Y_pred)
# print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# print("\n Classification Report:")
# print(classification_report(Y_test, Y_pred, target_names=iris.target_names))

# #VISUALISE THE CONFUSION MATRIXl
# cm = confusion_matrix(Y_test, Y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
# plt.xlabel('predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix - Naive Bayes')
# plt.show()

#KNN Algorithm [for both classification and regression]
# not suitable when there are outliers and imbalanced data

#1:IMPORTING DATASETS
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix

# #2:LOAD DATASETS
# iris = load_iris()
# X = iris.data #Features
# print(X)
# print()
# print()
# Y = iris.target #Target labels
# print(Y)
# print()
# print()

# #3:TRAIN TEST SPLIT(80 percent train, and 20% test)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# #4:FEATURE SCALING(ESSENTIAL FOR DISTANCE-BASED ALGORITHM)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# #5:MODEL TRAINING:
# #Using k =3 for this example
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)

# # 6: Prediction
# Y_pred = knn.predict(X_test)
# print(Y_pred)
# print()
# print()
# print(Y_test)
# print()
# print()
# # 7: Accuracy
# accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")


#Output:
# [5.  3.6 1.4 0.2]
#  [5.4 3.9 1.7 0.4]
#  [4.6 3.4 1.4 0.3]
#  [5.  3.4 1.5 0.2]
#  [4.4 2.9 1.4 0.2]
#  [4.9 3.1 1.5 0.1]
#  [5.4 3.7 1.5 0.2]
#  [4.8 3.4 1.6 0.2]
#  [4.8 3.  1.4 0.1]
#  [4.3 3.  1.1 0.1]
#  [5.8 4.  1.2 0.2]
#  [5.7 4.4 1.5 0.4]
#  [5.4 3.9 1.3 0.4]
#  [5.1 3.5 1.4 0.3]
#  [5.7 3.8 1.7 0.3]
#  [5.1 3.8 1.5 0.3]
#  [5.4 3.4 1.7 0.2]
#  [5.1 3.7 1.5 0.4]
#  [4.6 3.6 1.  0.2]
#  [5.1 3.3 1.7 0.5]
#  [4.8 3.4 1.9 0.2]
#  [5.  3.  1.6 0.2]
#  [5.  3.4 1.6 0.4]
#  [5.2 3.5 1.5 0.2]
#  [5.2 3.4 1.4 0.2]
#  [4.7 3.2 1.6 0.2]
#  [4.8 3.1 1.6 0.2]
#  [5.4 3.4 1.5 0.4]
#  [5.2 4.1 1.5 0.1]
#  [5.5 4.2 1.4 0.2]
#  [4.9 3.1 1.5 0.2]
#  [5.  3.2 1.2 0.2]
#  [5.5 3.5 1.3 0.2]
#  [4.9 3.6 1.4 0.1]
#  [4.4 3.  1.3 0.2]
#  [5.1 3.4 1.5 0.2]
#  [5.  3.5 1.3 0.3]
#  [4.5 2.3 1.3 0.3]
#  [4.4 3.2 1.3 0.2]
#  [5.  3.5 1.6 0.6]
#  [5.1 3.8 1.9 0.4]
#  [4.8 3.  1.4 0.3]
#  [5.1 3.8 1.6 0.2]
#  [4.6 3.2 1.4 0.2]
#  [5.3 3.7 1.5 0.2]
#  [5.  3.3 1.4 0.2]
#  [7.  3.2 4.7 1.4]
#  [6.4 3.2 4.5 1.5]
#  [6.9 3.1 4.9 1.5]
#  [5.5 2.3 4.  1.3]
#  [6.5 2.8 4.6 1.5]
#  [5.7 2.8 4.5 1.3]
#  [6.3 3.3 4.7 1.6]
#  [4.9 2.4 3.3 1. ]
#  [6.6 2.9 4.6 1.3]
#  [5.2 2.7 3.9 1.4]
#  [5.  2.  3.5 1. ]
#  [5.9 3.  4.2 1.5]
#  [6.  2.2 4.  1. ]
#  [6.1 2.9 4.7 1.4]
#  [5.6 2.9 3.6 1.3]
#  [6.7 3.1 4.4 1.4]
#  [5.6 3.  4.5 1.5]
#  [5.8 2.7 4.1 1. ]
#  [6.2 2.2 4.5 1.5]
#  [5.6 2.5 3.9 1.1]
#  [5.9 3.2 4.8 1.8]
#  [6.1 2.8 4.  1.3]
#  [6.3 2.5 4.9 1.5]
#  [6.1 2.8 4.7 1.2]
#  [6.4 2.9 4.3 1.3]
#  [6.6 3.  4.4 1.4]
#  [6.8 2.8 4.8 1.4]
#  [6.7 3.  5.  1.7]
#  [6.  2.9 4.5 1.5]
#  [5.7 2.6 3.5 1. ]
#  [5.5 2.4 3.8 1.1]
#  [5.5 2.4 3.7 1. ]
#  [5.8 2.7 3.9 1.2]
#  [6.  2.7 5.1 1.6]
#  [5.4 3.  4.5 1.5]
#  [6.  3.4 4.5 1.6]
#  [6.7 3.1 4.7 1.5]
#  [6.3 2.3 4.4 1.3]
#  [5.6 3.  4.1 1.3]
#  [5.5 2.5 4.  1.3]
#  [5.5 2.6 4.4 1.2]
#  [6.1 3.  4.6 1.4]
#  [5.8 2.6 4.  1.2]
#  [5.  2.3 3.3 1. ]
#  [5.6 2.7 4.2 1.3]
#  [5.7 3.  4.2 1.2]
#  [5.7 2.9 4.2 1.3]
#  [6.2 2.9 4.3 1.3]
#  [5.1 2.5 3.  1.1]
#  [5.7 2.8 4.1 1.3]
#  [6.3 3.3 6.  2.5]
#  [5.8 2.7 5.1 1.9]
#  [7.1 3.  5.9 2.1]
#  [6.3 2.9 5.6 1.8]
#  [6.5 3.  5.8 2.2]
#  [7.6 3.  6.6 2.1]
#  [4.9 2.5 4.5 1.7]
#  [7.3 2.9 6.3 1.8]
#  [6.7 2.5 5.8 1.8]
#  [7.2 3.6 6.1 2.5]
#  [6.5 3.2 5.1 2. ]
#  [6.4 2.7 5.3 1.9]
#  [6.8 3.  5.5 2.1]
#  [5.7 2.5 5.  2. ]
#  [5.8 2.8 5.1 2.4]
#  [6.4 3.2 5.3 2.3]
#  [6.5 3.  5.5 1.8]
#  [7.7 3.8 6.7 2.2]
#  [7.7 2.6 6.9 2.3]
#  [6.  2.2 5.  1.5]
#  [6.9 3.2 5.7 2.3]
#  [5.6 2.8 4.9 2. ]
#  [7.7 2.8 6.7 2. ]
#  [6.3 2.7 4.9 1.8]
#  [6.7 3.3 5.7 2.1]
#  [7.2 3.2 6.  1.8]
#  [6.2 2.8 4.8 1.8]
#  [6.1 3.  4.9 1.8]
#  [6.4 2.8 5.6 2.1]
#  [7.2 3.  5.8 1.6]
#  [7.4 2.8 6.1 1.9]
#  [7.9 3.8 6.4 2. ]
#  [6.4 2.8 5.6 2.2]
#  [6.3 2.8 5.1 1.5]
#  [6.1 2.6 5.6 1.4]
#  [7.7 3.  6.1 2.3]
#  [6.3 3.4 5.6 2.4]
#  [6.4 3.1 5.5 1.8]
#  [6.  3.  4.8 1.8]
#  [6.9 3.1 5.4 2.1]
#  [6.7 3.1 5.6 2.4]
#  [6.9 3.1 5.1 2.3]
#  [5.8 2.7 5.1 1.9]
#  [6.8 3.2 5.9 2.3]
#  [6.7 3.3 5.7 2.5]
#  [6.7 3.  5.2 2.3]
#  [6.3 2.5 5.  1.9]
#  [6.5 3.  5.2 2. ]
#  [6.2 3.4 5.4 2.3]
#  [5.9 3.  5.1 1.8]]


# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]


# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]


# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]


# Accuracy: 100.00%

#Decision Tree Algorithm(UNSUPERVISED ALGORITHM)
import pandas as pf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#LOADING THE DATASET
iris = load_iris()
X = iris.data #sepal/petal dimensions
Y = iris.target #flower species(0,1,2)
print(iris)
print()
print()
print(X)
print()
print()
print(Y)
print()
print()

#TRAINING AND TEST DATA SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#MODEL TRAINING:
#criterion='gini; (default) or we can use 'entropy' 
#max_depth= keeping it as 3 so that overfitting doesn;t happen
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, Y_train)

#6: PREDICTION:
Y_pred = clf.predict(X_test)
print(Y_pred)
print()
print()
print(Y_test)
print()
print()


#7: ACCURACY:
print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred)*100:.2f}%")
print("\n Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\n Detailed Report:\n", classification_report(Y_test, Y_pred))

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))

plot_tree(clf,
          feature_names= iris.feature_names,
          class_names=iris.target_names,
          filled = True,
          rounded= True,
          fontsize=12)

plt.title("Decision Tree Visualiation (Iris Dataset)")
plt.show()

