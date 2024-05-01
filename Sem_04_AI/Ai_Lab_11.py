# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# iris = pd.read_csv("G:\Course notes\Iris.csv")
# iris_df = iris.drop(columns='Species')

# scaler = StandardScaler()
# iris_scaled = scaler.fit_transform(iris_df)

# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(iris_scaled)
# iris_df['cluster'] = kmeans.labels_

# pca = PCA(n_components=2)
# iris_pca = pca.fit_transform(iris_scaled)
# iris_pca_df = pd.DataFrame(data=iris_pca, columns=['PC1', 'PC2'])

# plt.figure(figsize=(10, 6))

# colors = ['blue', 'red', 'green']
# for cluster, color in zip(range(3), colors):
#     cluster_points = iris_pca_df[iris_df['cluster'] == cluster]
#     plt.scatter(cluster_points['PC1'], cluster_points['PC2'], c=color, label=f'Cluster {cluster}')

# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Centroids')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('K-Means Clustering of Iris Dataset')
# plt.legend()
# plt.grid(True)
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine_df)

pca = PCA()
wine_pca = pca.fit_transform(wine_scaled)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), 
         marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

cumulative_variance = pca.explained_variance_ratio_.cumsum()
optimal_components = next(i for i, var in enumerate(cumulative_variance) if var >= 0.95) + 1
print("Optimal number of components:", optimal_components)

pca = PCA(n_components=optimal_components)
wine_pca_optimal = pca.fit_transform(wine_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(wine_pca_optimal[:, 0], wine_pca_optimal[:, 1], c=wine.target, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('First Two Principal Components of Wine Dataset')
plt.colorbar(label='Wine Class')
plt.grid(True)
plt.show()


