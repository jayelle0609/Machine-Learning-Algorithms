## UNSUPERVISED LEARNING

In the previous documents, we’ve discussed Supervised Learning (Regression & Classification). We’ve learned about learning from “labelled” data. There were already correct answers and our job back then was to learn how to arrive at those answers and apply the learning to new data.

But in this post it will be different. That’s because we’ll be starting with Unsupervised Learning wherein there were no correct answers or labels given. In other words, there’s only input data but there’s no output. There’s no supervision when learning from data.

In fact, Unsupervised Learning is algorithms are left on their own to discover things from data.
This is especially the case in Clustering wherein the goal is to reveal organic aggregates or “clusters” in data.

We just have a dataset and our goal is to see the groupings that have organically formed.
We’re not trying to predict an outcome here. The goal is to look for structures in the data. In other words, we’re “dividing” the dataset into groups wherein members have some similarities or proximities. For example, each ecommerce customer might belong to a particular group (e.g. given their income and spending level). If we have gathered enough data points, it’s likely there are aggregates.

At first the data points will seem scattered (no pattern at all). But once we apply a Clustering algorithm, the data will somehow make sense because we’ll be able to easily visualize the groups or clusters. Aside from discovering the natural groupings, Clustering algorithms may also reveal outliers for Anomaly Detection.

There are no rules set in stone when it comes to determining the number of clusters and which data point should belong to a certain cluster. It’s up to our objective (or if the results are useful
enough). This is also where our expertise in a particular domain comes in. Even with the most advanced tools and techniques, the context and objective are still crucial in making sense of data.

## K-Means Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

dataset = pd.read_csv('Mall_Customers.csv') 
dataset.head(10)

From sklearn.cluster import KMeans
wcss = [ ]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X) 
plt.scatter(X[y_kmeans == 0, 0],
X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue',
label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green',
label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan',
label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c =
'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =
300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

## Association Rule Learning

## Apriori

## Reinforcement Learning

## Artificial Neural Networks

(To be continued)
