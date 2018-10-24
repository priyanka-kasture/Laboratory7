# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:59:22 2018
@author: Priyanka Kasture
"""

# K-Means Clustering
# Importing the libraries
# import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

'''
x = [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72]
y = [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
'''
data = [[12,39],[20,36],[28,30],[18,52],[29,54],[33,46],[24,55],[45,59],[45,63],
        [52,70],[51,66],[52,63],[55,58],[53,23],[55,14],[61,8],[64,19],[69,7],[72,24]]

X = (pd.DataFrame(data,columns=['x','y'])).values
print(X)


# Using the elbow method to find the optimal number of clusters
'''
The Elbow method is a method of interpretation and validation of consistency within 
cluster analysis designed to help finding the appropriate number of clusters in a dataset.
'''
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method:')
plt.xlabel('Number of Clusters:')
plt.ylabel('WCSS')
plt.show()

'''
From the plot it is clear that the appropriate number of clusters is 3. Maximum = 4.
'''

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Centroids')

plt.title('Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
