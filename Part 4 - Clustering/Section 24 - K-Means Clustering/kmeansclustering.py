# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% imports
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.chdir('/Users/denismariano/pcloud/LEARNING/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering')

#%%
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#%% usind the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [] #within cluster sum of squares
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Weight of cluster square s')
plt.show()
    
#%% applying KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=5,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#%% visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Targed')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow', label='centroid')
plt.title('Clusters of clients')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending score (1-100)')

plt.legend()
plt.show()