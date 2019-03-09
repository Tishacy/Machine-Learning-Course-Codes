"""
	K-Means Cluster
	2018/8/18
	Tishacy
"""
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import time

"""
	Given a dataset without labels, divide the dataset into K clusters.
"""


def get_dataset(m=300):
    SEED = 23456
    rdm = np.random.RandomState(SEED)
    X_cluster_1 = 0.7 * rdm.randn(m // 2, 2) + 1
    X_cluster_2 = 0.7 * rdm.randn(m // 2, 2) + 3
    X = np.r_[X_cluster_1, X_cluster_2]
    rdm.shuffle(X)
    #print(X)
    return X


def distance(x1, x2):
    try:
        return np.sum((x1 - x2)**2, axis=1)
    except:
        return np.sum((x1 - x2)**2)


def random_choose(m, K):
    indexes = []
    while True:
        index = np.random.choice(m, 1)[0]
        if index not in indexes:
            indexes.append(index)
        if len(indexes) == K:
            break
    return indexes


def K_means_per_times(X, K):
    m = len(X)	# number of the sample data
    n = len(X[0]) # number of the features
    # randomly pick up 2 sample datas to be initial cluster centers
    mu_index = random_choose(m, K)
    mus = np.array([X[index] for index in mu_index])
    iteration = 0
    while True:
        # Cluster assignment: calculate the distance between the mu (cluster center) and each sample data.
        # If the sample data is close to cluster k, label it to be k.
        
        # # Process using Iterations
        # labels = []
        # for i, data in enumerate(X):
        #     distances = [distance(data, mu) for k, mu in enumerate(mus)]
        #     labels.append(np.where(distances == np.min(distances))[0][0])

        # Process using Matrices
        labels = []
        distances = []
        for k, mu in enumerate(mus):
            distances.append(distance(X, mu*np.ones([m, n])))
        distances = np.transpose(distances)
        for i, dist in enumerate(distances):
            labels.append(np.where(dist == np.min(dist))[0][0])


        # Move cluster centers: calculate the mass center of the sample datas labeled k and move the mu_k 
        # (cluster center) to it.

        # # Process using Iterations
        # Clusters = [[] for i in range(K)]
        # for j, label in enumerate(labels):
        # 	Clusters[label].append(X[j])
        # last_mus = np.copy(mus)
        # for k in range(K):
        # 	mus[k] = np.mean(Clusters[k], axis=0)

        # Process using Matrices
        labels = np.transpose(np.array([labels])) * np.ones([m, n])
        last_mus = np.copy(mus)
        for k in range(K):
            labels_k = (labels==k)
            mus[k] = np.sum(X * labels_k, axis=0) / (len(np.where(labels_k==True)[0])/n)
        labels = np.transpose(labels[:,0])

        #print("\niteration %d:" %iteration)
        #print('mus: ', mus)
        iteration += 1
        
        if (last_mus == mus).all():
        	break


    # Distortion function J
    J = np.mean([distance(data, mu)  for mu in mus  for data in X])
    return labels, mus, J


def K_means(X, K, times):
    Labels = []
    Mus = []
    Js = []
    for i in range(times):
        #print("====== Running the K-Means Algorithm for %d times ======" %(i+1))
        labels, mus, J = K_means_per_times(X, K)
        Labels.append(labels)
        Mus.append(mus)
        Js.append(J)
        print("the %d times, J = %.6f" %(i+1, J))
    opt_index = np.where(Js==np.min(Js))[0][0]
    labels = Labels[opt_index]
    mus = Mus[opt_index]
    J = Js[opt_index]
    return labels, mus, J


def test():
    X = get_dataset(300)
    #X = make_moons(300, noise=0.30)[0]
    print("Already get dataset")
    time0 = time.time()
    labels, mus, J = K_means(X, 2, 5)
    print("\nOptimized J = %.6f, costs %f s" %(J, time.time() - time0))
    print("Clusters' coordinates:\n", mus)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=8, c='k', alpha=0.7, label='unlabeled sample data')
    plt.title('Unlabled Sample Data')
    plt.legend()


    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], c=labels, s=8, cmap='jet', label='sample data')
    plt.scatter(mus[:,0], mus[:,1], c='k', label="cluster centers")
    plt.title('Labeled Sample Data')
    plt.legend()

    plt.show()

"""
Test: 300 sample data points withou 2 clusters for 100 times

Without numerical optimization:
>>> Optimized J = 4.910132, costs 24.160221 s
>>> Clusters' coordinates:
 [[0.97862753 1.11267062]
 [3.08627186 2.98693656]]

After numerical optimization:
>>> Optimized J = 4.910132, costs 6.402558 s
>>> Clusters' coordinates:
 [[0.97862753 1.11267062]
 [3.08627186 2.98693656]]

"""
if __name__=="__main__":
    test()