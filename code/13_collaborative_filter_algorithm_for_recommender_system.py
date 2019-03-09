"""
	13 Collaborative filter algorithm for recommender system
	
	Given the ratings of the m products by nu users (empty rating included), we want to establish a model to learn the performance of the products and the users' personal preferences, therefore we can recommend some products for the specific user.

	This problem could be demonstrated as follows:
		
	1.Suppose that the ratings of m products by nu nusers is Y:

				Y 	 	user_1	 user_2  ...   user_nu
			product_1	 y_11	  y_12	 ...    y_1nu
			product_2	 y_21     y_22   ...    y_2nu
			   ...		 ...	  ...	 ...     ...	
			product_m	 y_m1 	  y_m2   ...    y_mnu
		
		so that the dimension of Y is m × nu.
	
	2.Suppose that the performance of m products is described as X, and there are n features to describe a product:

				X 		feature_1  feature_2  ...  feature_n
			product_1	  x_11		 x_12     ... 	  x_1n
			product_2	  x_12 		 x_22	  ...     x_2n
			   ...		  ...		 ...	  ...	  ...
			product_m	  x_m1		 x_m2	  ...     x_mn
		
		so that the dimension of X is m × n
	
	3.Suppose that the users' preferences about the n features is Theta, and there are nu users.

			   Theta      user_1	 user_2    ...   user_nu
			feature_1	 theta_11	theta_12   ...  theta_1nu
			feature_2	 theta_12  	theta_22   ...  theta_2nu
			   ...		    ...		  ... 	   ...     ...
			feature_n	 theta_1n 	theta_2n   ...  theta_nnu

		so that the dimension of Theta is n × nu

	4.The problem is:
		Given Y, try to get the proper Theta and X.

	To get the proper Theta and X, we are actually doing an optimization puzzle. Here the target function to be optimized is what we called as cost function J as follows:

		J = 1/2 .* ∑(theta^T .* x - y)^2 + lambda/2 .* ∑x.^2 + lambda/2 .* ∑theta.^2
		  = 1/2 .* ∑(X * Theta - Y)^2 + lambda/2 .* ∑x.^2 + lambda/2 .* ∑theta.^2

		δJ/δx_i = ∑(theta^T .* x - y).*theta_i + lambda.*x_i
		δJ/δtheta_i = ∑(theta^T .* x - y).*x_i + lambda.*theta_i
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_dataset():
	"""
		get the ratings of m products by nu nusers is Y (empty ratings included).
	"""
	Y_df = pd.read_csv('./movie_ratings/data.csv') 
	Y = Y_df.values[:,1:]
	return Y


def cost_func(Theta, X, Y):
	"""
		calculate the cost function J and its derivatives.
	"""
	lamda = 0	# regularization parameter

	diff = (np.dot(X, Theta) - Y)
	diff = pd.DataFrame(diff).fillna(0).values # replace the NaN to 0 to avoid its impacts on later sum.
	J = 1/2 * np.sum(diff**2) + lamda/2 * np.sum(X**2) + lamda/2 * np.sum(Theta**2)
	dev_J_X = np.dot(diff, np.transpose(Theta)) + lamda*X
	dev_J_Theta = np.dot(np.transpose(X), diff) + lamda*Theta
	return J, dev_J_X, dev_J_Theta


def train(Theta, X, Y, alpha=0.01):
	"""
		train the model using gradient descent algorithm
	"""
	# calculate the cost function J and its partial derivatives
	J, dev_J_X, dev_J_Theta = cost_func(Theta, X, Y)
	# initialization
	step = 0
	Js = [J]
	steps = [step]
	while True:
		print("\tstep = %d, J = %.6f" %(step, J))
		last_J = J
		# update the X and Theta
		X = X - alpha * dev_J_X
		Theta =  Theta - alpha * dev_J_Theta
		# calculate the cost function J and its partial derivatives
		J, dev_J_X, dev_J_Theta = cost_func(Theta, X, Y)
		# record
		step += 1
		Js.append(J)
		steps.append(step)
		if abs(last_J - J) < 1e-7:
			break
	print("X:\n", X)
	print("Y:\n", Y)
	print("Theta:\n", Theta)
	return Theta, X, steps, Js




# get the Y dataset
Y = get_dataset()

# initialize the X and Theta
m = len(Y)
nu = len(Y[0])
n = 2 	# number of features. set 2 for easier visualization.
X = np.random.uniform(0, 1, [m, n])
Theta = np.random.randn(n, nu)

# train the model using gradient descent algorithm
Theta, X, steps, Js = train(Theta, X, Y, alpha=0.001)

# test
H = np.dot(X, Theta)
H = np.array([[int(value+0.5) for row in H for value in row]]).reshape(H.shape)
print("Hypothese:\n", H)

# recommend products for each user
for i in range(nu):
	no_ratings_index = np.where(H[:,i]-Y[:,i]!=0)[0]
	recommend_ratings = H[:,i][no_ratings_index]
	info = {
		"index": no_ratings_index,
		"rating": recommend_ratings
	}
	recommends = pd.DataFrame(info).sort_values(by="rating", ascending=False).head(2)
	print("\nuser %d:\n" %i, recommends)


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(steps, Js, color='r', alpha=0.7, label="training performance")
plt.xlabel(r"No. of iterations")
plt.ylabel(r"cost function $J$")
plt.title('Training Performance')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.scatter(X[:,0], X[:,1], s=18, c='k', alpha=0.7, label='movie clusters')
plt.scatter(Theta[0], Theta[1], s=18, c='r', alpha=0.7, label='user clusters')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Movie and user Clusters')
plt.legend()
plt.grid(True)

plt.show()