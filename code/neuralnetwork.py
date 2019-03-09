"""
	NeralNetwork

	classes and functions required to build a neural network
"""

import numpy as np
import matplotlib.pyplot as plt


# the class of neural network
class NeuralNetwork(object):
	def __init__(self,
				 num_inputs,
				 num_hidden_layers,
				 num_nodes_per_hidden,
				 num_outputs):
		self.num_inputs = num_inputs
		self.num_hidden_layers = num_hidden_layers
		self.num_nodes_per_hidden = num_nodes_per_hidden
		self.num_outputs = num_outputs
		self.ini_Theta = self.ini_weights(num_inputs, num_hidden_layers, num_nodes_per_hidden, num_outputs)


	def ini_weights(self, num_inputs, num_hidden_layers, num_nodes_per_hidden, num_outputs):
		"""
			returns the initialized the parameters of the neural network
		"""
		Theta = []
		#SEED = 23455
		#rdm = np.random.RandomState(SEED)
		for i in range(num_hidden_layers + 1):
			if i == 0:
				# the input layer to the first hidden layer
				theta = np.random.randn(num_inputs+1, num_nodes_per_hidden)
			elif i == num_hidden_layers:
				# the last hidden layer to the output layer
				theta = np.random.randn(num_nodes_per_hidden+1, num_outputs)
			else:
				theta = np.random.randn(num_nodes_per_hidden+1, num_nodes_per_hidden)
			Theta.append(theta)
		return Theta


	def forward_propagation(self, X, Theta, bias_value=1):
		"""
			implements forward propagation given the input vector/matrix of X,
			and return the hypothesis value h and the activated parameter matrices A
		"""
		a = X
		A = [a]
		for theta in Theta:
			a = self._add_bias(a, bias_value)
			a = self._sigmoid(np.dot(a, theta))
			A.append(a)
		h = a
		return h, A


	def _add_bias(self, nodes, bias_value=1):
		"""
			Add the bias node of given layer nodes.
			Note: Here the 'ith layer nodes'(nodes) is mathematically the activated matrix a
				of (i-1)th layer to ith layer. If i == 0, which means the given layer is the input layer, the 'nodes'
				is mathematically the input matrix X. 
		"""
		m = len(nodes)
		bias_node = bias_value * np.ones([m, 1])	# define a bias node
		return np.c_[bias_node, nodes]
		

	def _drop_bias(self, nodes, bias_value=1):
		"""
			Drop the bias node of given layer nodes.
			Note: Here the 'ith layer nodes'(nodes) is mathematically the activated matrix a
				of (i-1)th layer to ith layer. If i == 0, which means the given layer is the input layer, the 'nodes'
				is mathematically the input matrix X. 
		"""
		return np.delete(nodes, 0, axis=1)


	def _sigmoid(self, z):
		"""
			Sigmoid activation function
			It will return the activated matrix a of ith layer (a_i) while given the z of (i-1)th layer (z_(i-1)).
				
				a_i = _sigmoide(z_(i-1)) = 1 / (1 + exp(-z_(i-1)))
				z_(i-1) = a_(i-1) * theta_(i-1)

			where	a_i is the activated matrix of ith layer
					theta_i is the weights (parameters) of the ith layer
		"""
		return 1 / (1 + np.exp(-z))


	def cost_function(self, Y, h, Theta, lamda=1):
		"""
			Cost function J
			It will return the cost function values, given output label matrix (Y), hypothesis matrix (h),
			weights of all layers (Theta) and the regularization parameter (lamda λ, optional)
		"""
		m = len(Y)
		J = -1 / m * np.sum(Y*np.log(h) + (1-Y)*np.log(1-h)) + lamda / (2*m) * np.sum([np.sum(theta**2) for theta in Theta])
		return J


	def gradient_descent(self, X, Y, ini_Theta, alpha=0.01, lamda=1, exponential_decay=False, threshold=1e-7):
		"""
			implement the gradient descent algorithm with back propagation, and returns the
			optimized weights (parameters) of all layers (Theta), times of iterations (steps),
			and the cost function values (J)
		"""
		# forward propagation and calculate the J
		h, A = self.forward_propagation(X, ini_Theta, bias_value=1)
		J = self.cost_function(Y, h, ini_Theta, lamda)
		# initialization of the algorithm
		#Theta = np.array(ini_Theta)
		Theta = np.array([theta.copy() for theta in ini_Theta])
		last_J = J
		step = 0
		Js = [J]
		steps = [step]
		Delta = np.array([theta.copy() for theta in Theta])	# just used to set Delta to have the same dimension as Theta 
		num_interval = len(Theta)	# number of intervals between layers
		# run gradient descent algorithm
		print("===== Running Gradient Descent Algorithm with Back Propagation =====")
		while True:
			DD = []
			last_J = J
			# back propatation (BP)
			if step % 50 == 0:
				print("    step: %d, J: %.6f, α: %.6f" %(step, J, alpha))
			if exponential_decay == True:
				# exponential decay leraning rate
				alpha = alpha * 0.99**(step/5) + 1e-6
			for i in range(num_interval):
				if i == 0:
					# BP of the last layer
					Delta[-1] = h - Y
				else:
					Delta[num_interval-(i+1)] = self._drop_bias(np.dot(Delta[num_interval-i], np.transpose(Theta[num_interval-i])) * self._add_bias(A[num_interval-i]) * (1 - self._add_bias(A[num_interval-i])))
				# calculate the D (∂J/∂θ)
				D = np.dot(np.transpose(self._add_bias(A[num_interval-(i+1)])), Delta[num_interval-(i+1)])
				DD.append(D)
				# update the theta values
				Theta[num_interval-(i+1)] = Theta[num_interval-(i+1)] - alpha * D
			# forward propagation and calculate J
			h, A = self.forward_propagation(X, Theta, bias_value=1)
			J = self.cost_function(Y, h, Theta, lamda)
			# record results
			Js.append(J)
			steps.append(step)
			step += 1

			# end condition
			if abs(last_J - J) < threshold or step >= 30000:
				break
		print("===== Fitting Over =====")
		print("    J = %.6f" %J)
		print("    steps = %d" %step)
		return Theta, steps, Js



# learning curve
def learning_curve(nn, scale_hidden_layers, scale_lamda, num_slices,
				   X_dataset, Y_dataset, alpha=0.0001, exponential_decay=False):
	"""
		Learning curvs are used to optimize the neural network.
	"""
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.figure(figsize=(12, 3.6))
	
	# divide the dataset into 3 parts: train set, cross validation set
	# and test set
	m = len(X_dataset)
	X_train = X_dataset[:int(0.6*m)]	# 60%
	X_cv = X_dataset[int(0.6*m):int(0.8*m)]	# 20% 
	X_test = X_dataset[int(0.8*m):]		# 20%

	Y_train = Y_dataset[:int(0.6*m)]	# 60%
	Y_cv = Y_dataset[int(0.6*m):int(0.8*m)]	# 20% 
	Y_test = Y_dataset[int(0.8*m):]		# 20%

	# learning curve corresponding to number of hidden layers
	num_inputs = nn.num_inputs
	num_outputs = nn.num_outputs
	num_nodes_per_hidden = nn.num_nodes_per_hidden
	J_train_hl = []
	J_cv_hl = []
	for num_hidden_layers in scale_hidden_layers:
		print("       %d Hidden Layers" %num_hidden_layers)
		NN = NeuralNetwork(num_inputs, num_hidden_layers, num_nodes_per_hidden, num_outputs)
		# train the neural network (nn)
		Theta, steps, Js = NN.gradient_descent(X_train, Y_train, NN.ini_Theta, alpha=alpha, lamda=0, exponential_decay=exponential_decay)
		# J of train set 
		h_train = NN.forward_propagation(X_train, Theta)[0]
		J_train = NN.cost_function(Y_train, h_train, Theta, lamda=0)
		J_train_hl.append(J_train)
		# J of cross validation
		h_cv = NN.forward_propagation(X_cv, Theta)[0]
		J_cv = NN.cost_function(Y_cv, h_cv, Theta, lamda=0)
		J_cv_hl.append(J_cv)
	print("J_train:", J_train_hl)
	print("J_cv", J_cv_hl)

	plt.subplot(131)
	plt.plot(scale_hidden_layers, J_train_hl, '-o', alpha=0.7, label=r'$J_{train}$')
	plt.plot(scale_hidden_layers, J_cv_hl, '-o', alpha=0.7, label=r'$J_{cv}$')
	plt.xlabel(r'$N_{hidden}$')
	plt.ylabel(r"cost function $J$")
	plt.title(r"$N_{hidden}$")
	plt.legend()


	# learning curve corresponding to lamda
	J_train_ld = []
	J_cv_ld = []
	for lamda in scale_lamda:
		print("       lamda = %.6f" %lamda)
		# train the neural network (nn)
		Theta, steps, Js = nn.gradient_descent(X_train, Y_train, nn.ini_Theta, alpha=alpha, lamda=lamda, exponential_decay=exponential_decay)
		# J of train set 
		h_train = nn.forward_propagation(X_train, Theta)[0]
		J_train = nn.cost_function(Y_train, h_train, Theta, lamda=lamda)
		J_train_ld.append(J_train)
		# J of cross validation
		h_cv = nn.forward_propagation(X_cv, Theta)[0]
		J_cv = nn.cost_function(Y_cv, h_cv, Theta, lamda=lamda)
		J_cv_ld.append(J_cv)
	print("J_train:", J_train_ld)
	print("J_cv", J_cv_ld)

	plt.subplot(132)
	plt.plot(scale_lamda, J_train_ld, '-o', alpha=0.7, label=r'$J_{train}$')
	plt.plot(scale_lamda, J_cv_ld, '-o', alpha=0.7, label=r'$J_{cv}$')
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r"cost function $J$")
	plt.title(r"$\lambda$")
	plt.legend()
	

	# learning curve corresponding to size of dataset
	J_train_ds = []
	J_cv_ds = []
	size_of_trainset = []
	for i in range(num_slices):
		X_train_slice = X_train[:int((i+1)/ num_slices *len(X_train))]
		Y_train_slice = Y_train[:int((i+1)/ num_slices *len(Y_train))]
		size_of_trainset.append(len(X_train_slice))
		print("       size of train set: %d" %len(X_train_slice))
		# train the neural network (nn)
		Theta, steps, Js = nn.gradient_descent(X_train_slice, Y_train_slice, nn.ini_Theta, alpha=alpha, lamda=0, exponential_decay=exponential_decay)
		# J of train set 
		h_train = nn.forward_propagation(X_train_slice, Theta)[0]
		J_train = nn.cost_function(Y_train_slice, h_train, Theta, lamda=0)
		J_train_ds.append(J_train)
		# J of cross validation
		h_cv = nn.forward_propagation(X_cv, Theta)[0]
		J_cv = nn.cost_function(Y_cv, h_cv, Theta, lamda=0)
		J_cv_ds.append(J_cv)
	print("J_train:", J_train_ds)
	print("J_cv", J_cv_ds)

	plt.subplot(133)
	plt.plot(size_of_trainset, J_train_ds, '-o', alpha=0.7, label=r'$J_{train}$')
	plt.plot(size_of_trainset, J_cv_ds, '-o', alpha=0.7, label=r'$J_{cv}$')
	plt.xlabel(r'$N_{samples}$')
	plt.ylabel(r"cost function $J$")
	plt.title(r"$N_{samples}$")
	plt.legend()

	plt.tight_layout()
	







#########################################################
#														#
#						A Test							#
#														#
#########################################################			

def get_dataset():
	n = 2	# n features
	m = 3000	# m samples 
	SEED = 23455
	rdm = np.random.RandomState(SEED)
	X = rdm.uniform(0.0, 4.0, (n, m))
	# 将下面区域划分为4个
	y = np.ones(m)
	y = np.array(list(map(lambda x_0, x_1: [1, 0, 0, 0] if x_0**2+x_1**2 < 2 and x_1 < 2 and np.random.rand() < 0.8
						 else ([0, 1, 0, 0] if 2 <= x_0 and x_1 < 2 and np.random.rand() < 0.8
						   else ([0, 0, 1, 0] if x_0 < 2 and 2 <= x_1 and np.random.rand() < 0.8
							 else [0, 0, 0, 1])), X[0], X[1])))
	#print(y)
	return np.transpose(X), y


def _test():
	##### Define the Neural Network and Implement the Gradient Descent Algorithm #####
	A = NeuralNetwork(2, 2, 4, 4)
	print("number of nodes in the input layer: %d" %A.num_inputs, A.num_outputs, A.num_hidden_layers, A.num_nodes_per_hidden)
	print("number of hidden layers: %d" %A.num_hidden_layers)
	print("number of nodes per hidden layer: %d" %A.num_nodes_per_hidden)
	print("number of nodes in the output layer: %d" %A.num_outputs)
	X, Y = get_dataset()
	Theta, steps, Js = A.gradient_descent(X, Y, A.ini_Theta, alpha=0.0003, lamda=1)


	###### plot the result ######
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family'] = 'STIXGeneral'

	plt.plot(steps, Js, '-', color='r', alpha=0.5,
			 label=r"cost function $J$ decay")
	plt.xlabel(r'No. of iterations', fontsize=12)
	plt.ylabel(r'cost function $J$', fontsize=12)
	plt.legend()



	##### Test Value #####
	m = 1000
	X_ = np.random.normal(2.0, 1.0, (m, 2))

	# 获取结果数据
	labels = A.forward_propagation(X_, Theta)[0]
	# 将结果可视化
	plt.figure(figsize=(10, 5))

	colors_ = []
	for i, label in enumerate(labels):
		max_i = np.where(label==np.max(label))[0][0]
		#print(np.where(label==np.max(label))[0][0])
		if max_i == 0:
			colors_.append('indianred')
		elif max_i == 1:
			colors_.append('steelblue')
		elif max_i == 2:
			colors_.append('dimgrey')
		else:
			colors_.append('orchid')

	colors = []
	for i in range(len(Y)):
		max_i_y = np.where(Y[i]==np.max(Y[i]))[0][0]
		#print(np.where(Y[i]==np.max(Y[i]))[0][0])
		if max_i_y == 0:
			colors.append('indianred')
		elif max_i_y == 1:
			colors.append('steelblue')
		elif max_i_y == 2:
			colors.append('dimgrey')
		else:
			colors.append('orchid')
		
	#	colors_ = list(map(lambda x: 'indianred' if x==[1,0,0,0] else ('steelblue' if x==[0,1,0,0] else ('dimgrey' if x==[0,0,1,0] else 'orchid')), labels))
	#	colors = list(map(lambda x: 'indianred' if x==[1,0,0,0] else ('steelblue' if x==[0,1,0,0] else ('dimgrey' if x==[0,0,1,0] else 'orchid')), Y))
	plt.subplot(121)
	plt.scatter(X_[:,0], X_[:,1], color=colors_, s=18, label='test data')
	plt.xlabel(r"$feature_0$", fontsize=12)
	plt.ylabel(r"$feature_1$", fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend(loc='upper right')

	plt.subplot(122)
	plt.scatter(X[:,0], X[:,1], color=colors, s=18, label='trained data', alpha=0.3)
	plt.xlabel(r"$feature_0$", fontsize=12)
	plt.ylabel(r"$feature_1$", fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend(loc='upper right')
	plt.suptitle("Tests of the Model", fontsize=15)

	plt.show()





if __name__=="__main__":
	A = NeuralNetwork(2, 2, 4, 4)
	print("number of nodes in the input layer: %d" %A.num_inputs)
	print("number of hidden layers: %d" %A.num_hidden_layers)
	print("number of nodes per hidden layer: %d" %A.num_nodes_per_hidden)
	print("number of nodes in the output layer: %d" %A.num_outputs)
	X, Y = get_dataset()
	scale_hidden_layers = np.arange(1, 3)
	scale_lamda = np.arange(0, 2, 0.5)
	num_slices = 3
	# learning_curve(A, scale_hidden_layers, scale_lamda, num_slices, X, Y, alpha=0.0001, exponential_decay=False)

	# plt.show()
	_test()

