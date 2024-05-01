import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def assign_to_nearest_cluster(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = distances.index(min(distances))
        labels.append(cluster)
    return labels

def kmeans(data, k, max_iters=100):
    centroids = data[np.random.choice(len(data), k)]

    for _ in range(max_iters):
        labels = assign_to_nearest_cluster(data, centroids)

        new_centroids = np.array([data[np.array(labels) == i].mean(axis=0) for i in range(k)])

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100, 2)

    k = 3  # Number of clusters
    centroids, labels = kmeans(data, k)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.show()

# import matplotlib.pyplot as plt

# x = [10, 12, 8, 5, 3, 20, 15, 1]
# y = [3, 2, 6, 7, 4, 10, 12, 1]


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# data = list(zip(x, y))

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(data)

# plt.scatter(x, y, c=kmeans.labels_)
# plt.show()


        




