# Decision Tree Regression

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('/Users/denismariano/pcloud/LEARNING/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression')
dataset = pd.read_csv('Position_Salaries.csv')

#%% Importing the dataset
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#%% Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#%% Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#%% Predicting a new result
y_pred = regressor.predict(6.5)

#%% Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), .01) #increased resolution
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()