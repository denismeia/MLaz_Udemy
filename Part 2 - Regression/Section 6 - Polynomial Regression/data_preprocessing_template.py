# Data Preprocessing Template

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/denismariano/pcloud/udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression')
dataset = pd.read_csv('Position_Salaries.csv')

#%% Importing the dataset
#FEATURES have to be always a matrix
X = dataset.iloc[:, 1:2].values
# the dependent variable doesn't have to be a matrix
y = dataset.iloc[:, 2].values

#%% fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#%% fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#%% Linear regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X))

#%% Polynomial regression results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X))
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)))

#%% Polynomial regression results with a finer grid
X_grid = np.arange(min(X),max(X),.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid,lin_reg.predict(X_grid))
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)))

#%% predicting a new result with regression models
print(lin_reg.predict(6.5)) #Linear
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))