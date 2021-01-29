import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("/Users/nithinkyatham/Downloads/Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

# Elbow method using WCSS
from sklearn.cluster import KMeans
# Creating a list to hold WCSS values for different K Values
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plot the graph of WCSS values versus number of clusters to get elbow angle
"""plt.plot(range(1,11),wcss,color='blue')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS-Values')
plt.title('Graph of WCSS versus number of clusters')"""

# Implementing K-Means with K=5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init=10)

# Fitting and Predicting the clusters of each customer
y_kmeans = kmeans.fit_predict(X)

#Visual Representtion of the cluster-formation
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s = 100, color = 'red', label = 'Cluster 0')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s = 100, color = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s = 100, color = 'yellow', label = 'Cluster 2')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s = 100, color = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s = 100, color = 'cyan', label = 'Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300, c='black', label='Centroids')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of customers')
plt.legend()

