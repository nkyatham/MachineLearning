import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("/Users/nithinkyatham/Downloads/Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

# Creating the Dendrogram to judge the ideal number of groups
"""from scipy.cluster import hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward', metric = 'euclidean'))
plt.title('Dendrogram for the customer-grouping')
plt.xlabel('Customers')
plt.ylabel('Distance using Euclidean')"""

# Implementing Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering as ac

hc = ac(n_clusters=5, affinity='euclidean', linkage='ward')

# Fitting and Predicting the clusters of each customer
y_hc = hc.fit_predict(X)

#Visual Representtion of the cluster-formation
plt.scatter(X[y_hc ==0,0], X[y_hc ==0,1], s = 100, color = 'red', label = 'Cluster 0')
plt.scatter(X[y_hc ==1,0], X[y_hc ==1,1], s = 100, color = 'blue', label = 'Cluster 1')
plt.scatter(X[y_hc ==2,0], X[y_hc ==2,1], s = 100, color = 'yellow', label = 'Cluster 2')
plt.scatter(X[y_hc ==3,0], X[y_hc ==3,1], s = 100, color = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc ==4,0], X[y_hc ==4,1], s = 100, color = 'cyan', label = 'Cluster 4')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of customers')
plt.legend()



