# Data Preprocessing Template

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/denismariano/pcloud/udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression')

#%% Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#%% Fitting Linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#%% Predicting the train set results and plotting
y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train, color='red')
plt.scatter(X_test,y_test, color='green')

plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary x XP, training')
plt.xlabel('Years of XP')
plt.ylabel('Salary')
plt.show()

#%% visualizing test results
regressor.score(X_train,y_train)