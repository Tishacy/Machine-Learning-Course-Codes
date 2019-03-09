"""
	Principle Component Analysis (PCA)
	2018/8/20
	Tishacy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from k_means import *

"""
	Target:
		Compress the sample dataset X (m × n) to be Z (m × k),
		which is actually compress the n features into k features (k < n)

	step 1:
		Compute "covariance matrix":
			Sigma = 1/m * ∑^n (x .* x^T)
			 	  = 1/m * (X^T * X)
			where	X is the sample dataset (m × n)
					Sigma is the matrix (n × n)
	step 2:
		Compute "eigenvectors" of matrix Sigma:
			[U, S, V] = svd(Sigma)
			where	svd() is the function to do Singular Value Decomposition
					U is the mu matrix (n × n)
					S is the diagonal matrix of sigma (n × n)
	step 3:
		Determine the number of features k to be compressed:
		1. choose k (the dimension to be compression)
		2. check the following inequality:
			∑^k(s) ./ ∑^n(s) >= 0.99 
			which also could be written as:
				np.sum(S[:k]) / np.sum(S) >= 0.99
			if the inequaity is True:
				return mus (the first k columns in U, (n × k))
			else:
				k += 1 and repeat 2. in step 3 
	step 4:
		Compute the compressed data Z (m × k):
			Z = X * U
"""

def get_dataset(m=300, n=5):
    SEED = 2
    rdm = np.random.RandomState(SEED)
    X_cluster_1 = 1 * rdm.randn(m // 2, n) + 1
    X_cluster_2 = 1 * rdm.randn(m // 2, n) + 3
    X = np.r_[X_cluster_1, X_cluster_2]
    rdm.shuffle(X)
    #print(X)
    return X


def PCA(X, precision=0.95):
	m = len(X)
	# Compute Sigma
	Sigma = 1/m * np.dot(np.transpose(X), X)
	# Singular values decomposition
	[U, S, V] = np.linalg.svd(Sigma)
	# Determine k
	k = 1
	while np.sum(S[:k])/np.sum(S) < precision:
		k += 1
	mus = U[:, :k]
	# Compute Z
	Z = np.dot(X, mus)
	return Z, mus, k



X = get_dataset(300, 3)
n = len(X[0])
[Z, U, k] = PCA(X, 0.95)
plot_min = np.min(Z[:,0])
plot_max = np.max(Z[:,0])

print("Compress %d features to be %d features" %(n, k))
print('Vectors (mus):')
print(U)
plt.figure(figsize=(8.5, 4))

ax = plt.subplot(121, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c='k', s=18, alpha=0.7)
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
ax.set_zlabel(r'$x_2$')
plt.title('Uncompressed Dataset')

plt.subplot(122)
plt.scatter(Z[:,0], Z[:,1], c='k', s=18, alpha=0.7)
plt.xlabel(r'$z_0$')
plt.ylabel(r'$z_1$')
plt.xlim(plot_min, plot_max)
plt.ylim(-4, 4)
plt.title('Compressed Dataset')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)


# Run K_means Algorithm
labels, mus, J = K_means(Z, 2, 10)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(Z[:,0], Z[:,1], s=18, c='k', alpha=0.7, label='unlabeled sample data')
plt.title('Unlabled Sample Data')
plt.xlim(plot_min, plot_max)
plt.ylim(-4, 4)
plt.legend()

plt.subplot(122)
plt.scatter(Z[:,0], Z[:,1], c=labels, s=18, cmap='jet', label='sample data')
plt.scatter(mus[:,0], mus[:,1], c='k', label="cluster centers")
plt.title('Labeled Sample Data')
plt.xlim(plot_min, plot_max)
plt.ylim(-4, 4)
plt.legend()

plt.show()
