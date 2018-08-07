"""
A custom implementation of the K-Means unsupervised classification algorithm
Made with guidance from https://github.com/llSourcell/k_means_clustering
"""

import numpy as np
from matplotlib import pyplot as plt

def load_dataset(name):
    return np.loadtxt(name)

def distance(a, b):
    return np.linalg.norm(a - b)

def kmeans(dataset, k, epsilon=0):
    data_count, feature_count = dataset.shape

    #Chooses random points to be the centroids
    centroids = dataset[np.random.randint(0, data_count - 1, size=k)]
    centroids_old = np.zeros(centroids.shape)

    #The distance between the old centroids and the new centroids
    error = distance(centroids, centroids_old)

    #where each point is assigned to
    assignments = np.zeros((data_count, 1))
    while error > epsilon:
        centroids_old = centroids
        for point_index, dataPoint in enumerate(dataset):
            #Will store the distance to each centroid
            distance_to_centroids = np.zeros((k, 1))

            for centroid_index, centroid in enumerate(centroids):
                distance_to_centroids[centroid_index] = distance(centroid, dataPoint)

            assignments[point_index, 0] = np.argmin(distance_to_centroids)

        tmp_centroids = np.zeros(centroids.shape)
        for index in range(len(centroids)):
            closestPoints = [i for i in range(len(assignments)) if assignments[i] == index]
            #Setting axis tells numpy to take the mean over the columns
            new_centroid = np.mean(dataset[closestPoints], axis = 0)
            tmp_centroids[index] = new_centroid

        centroids = tmp_centroids
        error = distance(centroids_old, centroids)
    return centroids, assignments

def showPlot(dataSet, centroids, assignments):
    colors = ['r', 'g']
    for point_index, dataPoint in enumerate(dataSet):
        label = int(assignments[point_index, 0])
        plt.scatter(dataPoint[0], dataPoint[1], s=120, marker=".", color=colors[label])

    for centroid_index, centroid in enumerate(centroids):
        for label in range(len(centroids)):
            plt.scatter(centroid[0], centroid[1], s=120, marker="+", color= colors[label])
    plt.show()

dataset = load_dataset('durudataset.txt')
centroids, assignments = kmeans(dataset, 2)

showPlot(dataset, centroids, assignments)
