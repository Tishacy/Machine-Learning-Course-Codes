"""
	Back propagation 反向传播算法
	2018/8/14
	Tishacy
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Description:

	Hypothese function:
		In neural network, the hypothese function value(s) is/are the
		result(s) of the forward propagation. 

	Cost function J:
		Given that the cost function is only considering the difference
		between the hypothese function (the output layer) and the y labels,
		other than other layers, and the ouput node(s) is/are actually the 
		logistic regression classifier(s), the cost function J is pretty 
		similar to the logistic regression.
			None regularization:
				J(θ) = -1 / m * ∑^m ∑^K (y.*ln(h) + (1-y).*ln(1-h))
			Regularization:
				J(θ) = -1 / m * ∑^m ∑^K (y.*ln(h) + (1-y).*ln(1-h)) + λ /(2*m) * ∑(θ.^2)
	
	Partial derivative of J: D_i = (∂J/∂θ)_i   i means the ith layer
		D_i = (a_i^+)^T * δ_(i+1)^-
			where	a_i means the matrix of activated nodes' values
					^+ means adding the bias node
					^- means without the bias node
					δ_i^+ = δ_(i+1)^- * (θ_i)^T .* (a_i^+) .* (1 - a_i^+)
					δ_i^- = δ_i^+[1:] = drop_bias(δ_i^+)

	Gradient descent algorithm:
		θ_i := θ_i - α * D_i 
"""

# Define the training set
def get_trainset():
	m = 5000	# m means the number of samples
	n = 2	# n means the number of input nodes (num_feature) without bias
	k = 2	# k means the number of output nodes
	rdm = np.random.RandomState(23455)
	X = rdm.randn(m, n)
	#Y = np.array([list(map(lambda x: [0, 1] if X_[0]*X_[1]**2 < (X_[2]-X_[3])*X[4] else [1, 0], X_) for X_ in X)])
	Y = np.zeros([m, k])
	for i, X_ in enumerate(X):
		if X_[0]**2+X_[1]**3 <= 2:
		#if X_[0]<0.5:
			Y[i] = [1, 0]
		else:
			Y[i] = [0, 1]
	#print(Y)
	return X, Y


# Another dataset
def get_dataset():
	n = 2	# n features
	m = 1000	# m samples 
	X = np.random.uniform(0.0, 4.0, (n, m))
	# 将下面区域划分为4个
	y = np.ones(m)
	y = np.array(list(map(lambda x_0, x_1: [1, 0, 0, 0] if x_0**2+x_1**2 < 2 and x_1 < 2 
						 else ([0, 1, 0, 0] if 2 <= x_0 and x_1 < 2 
						   else ([0, 0, 1, 0] if x_0 < 2 and 2 <= x_1 
						   	 else [0, 0, 0, 1])), X[0], X[1])))
	#print(y)
	return np.transpose(X), y



# Define the neural network and randomly initialize it
# 	4 layers: 1 input layer, 2 hidden layers, 1 output layer
#	10 nodes per hidden layer (withou bias node)
#	5 input nodes, 2 output nodes
def neural_network():
	theta_0 = np.random.rand(2+1, 10)
	theta_1 = np.random.rand(10+1, 10)
	theta_2 = np.random.rand(10+1, 4)
	Theta = [theta_0, theta_1, theta_2]
	return Theta



############ functions needed ############
# Add bias node
def add_bias(nodes, bias_value=1):
	m = len(nodes)
	bias_node = bias_value * np.ones([m, 1])	# define a bias node
	return np.c_[bias_node, nodes]


# Drop bias node
def drop_bias(nodes):
	return np.delete(nodes, 0, axis=1)


# Sigmoid activation function 
def sigmoid(z):
	return 1 / (1 + np.exp(-z))


# Cost function J
def cost_function(Y, h, Theta, lamda=1):
	m = len(Y)
	J = -1 / m * np.sum(Y*np.log(h) + (1-Y)*np.log(1-h)) + lamda / (2*m) * np.sum([np.sum(theta**2) for theta in Theta])
	return J
############ functions needed ############



# Forward propagation
def forward_propagation_with_bias(X, Theta, bias_value=0.5):
	a = X
	A = [a]
	for theta in Theta:
		a = add_bias(a, bias_value)
		a = sigmoid(np.dot(a, theta))
		A.append(a)
	h = a
	return h, A


# Gradient descent algorithm
def gradient_descent(X, Y, Theta, alpha=0.1):
	# forward propagation and calculate the J
	h, A = forward_propagation_with_bias(X, Theta)
	J = cost_function(Y, h, Theta)
	# initialization
	last_J = J
	step = 0
	Js = [J]
	steps = [step]
	Delta = np.array(Theta)   # just used to set Delta to have the same dimension as Theta 
	num_interval = len(Theta)	# number of intervals between layers
	#print(num_interval)
	# gradient descent
	print("开始拟合")
	while True:
		# back propagation (BP)
		DD = []
		last_J = J
		#ND = gradient_check(Y, h, Theta) ###################################3
		if step % 100 == 0:
			print("step: %d, J: %.6f" %(step, J))
		for i in range(num_interval):
			if i == 0:
				# BP of the last layer
				Delta[-1] = h - Y
			else:
				#print(add_bias(A[num_interval-(i+1)]))
				Delta[num_interval-(i+1)] = drop_bias(np.dot(Delta[num_interval-i], np.transpose(Theta[num_interval-i])) * add_bias(A[num_interval-i]) * (1 - add_bias(A[num_interval-i])))
			# calculate the D (∂J/∂θ)
			D = np.dot(np.transpose(add_bias(A[num_interval-(i+1)])), Delta[num_interval-(i+1)])
			DD.append(D)
			# update the θ
			Theta[num_interval-(i+1)] = Theta[num_interval-(i+1)] - alpha * D
			#Theta[i] = Theta[i] - alpha * ND[i]	##############################
		# forward propagation and calculate J
		h, A = forward_propagation_with_bias(X, Theta)
		J = cost_function(Y, h, Theta)
		# record results
		Js.append(J)
		steps.append(step)

		# ##### gradient checking #####
		# if step == 0:
		# 	ND = gradient_check(Y, h, Theta)
		# 	DD = [DD[num_interval-(i+1)] for i in range(num_interval)]
		# 	for i, D in enumerate(DD):
		# 		nD = ND[i]
		# 		print("%d:\n" %i)
		# 		print(D)
		# 	break
		# ##### gradient checking over #####

		step += 1
		if abs(last_J - J) < 1e-6 or step >= 10000:
			break
	print("拟合结束")
	print("J = %.6f" %J)
	print("steps = %d" %step)
	#print("Theta:\n", Theta)
	return Theta, steps, Js



# Run the model
def run(alpha=0.001):
	X, Y = get_dataset()
	#X, Y = get_trainset()
	ini_Theta = neural_network()
	Theta, steps, Js = gradient_descent(X, Y, ini_Theta, alpha)
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family'] = 'STIXGeneral'
	
	plt.plot(steps, Js, '-', color='r', alpha=0.5,
			 label=r"cost function $J$ decay")
	plt.xlabel(r'No. of iterations', fontsize=12)
	plt.ylabel(r'cost function $J$', fontsize=12)
	plt.legend()

	# x = np.array([[1, 1.5]])
	# result = forward_propagation_with_bias(x, Theta)[0]
	# if result[0][0] > result[0][1]:
	# 	result_label = True
	# else:
	# 	result_label = False

	# x_ = x[0]
	# y = x_[0]**2+x_[1]**3 <= 2
	# print("input:\n    ", x)
	# print("output:\n    ", result, "  ", result_label)
	# print("correct answer:\n    ", y)
		# 生成m组数据作为测试集 X_
	m = 1000
	X_ = np.random.normal(2.0, 1.0, (m, 2))

	# 获取结果数据
	labels = forward_propagation_with_bias(X_, Theta)[0]
	# 将结果可视化
	plt.figure(figsize=(6, 5))
	
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
	plt.scatter(X_[:,0], X_[:,1], color=colors_, s=18, label='test data')
	plt.scatter(X[:,0], X[:,1], color=colors, s=18, label='trained data', alpha=0.3)
	plt.xlabel(r"$feature_0$", fontsize=12)
	plt.ylabel(r"$feature_1$", fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend(loc='upper right')
	plt.title("Tests of the Model", fontsize=15)

	plt.show()


# Numerical gradient checking
# Something wrong here...
def gradient_check(Y, h, Theta):
	ND = [theta.copy() for theta in Theta] #np.array(Theta).copy()
	epsilon = 1e-5
	#print("Theta:\n", Theta)
	for i, theta in enumerate(Theta):
		for j, theta_row in enumerate(theta):
			for k, theta_col in enumerate(theta_row):
				print("\n%d %d %d:" %(i, j, k))
				print("Theta:\n", Theta[i][j][k])

				Theta_add_epsilon =[theta.copy() for theta in Theta]
				Theta_add_epsilon[i][j][k] += epsilon
				Theta_minus_epsilon = [theta.copy() for theta in Theta]
				Theta_minus_epsilon[i][j][k] -= epsilon
				
				print("Theta_add_epsilon:\n", Theta_add_epsilon[i][j][k])
				print("Theta_minus_epsilon:\n", Theta_minus_epsilon[i][j][k])

				
				ND[i][j][k] = (cost_function(Y, h, Theta_add_epsilon) - cost_function(Y, h, Theta_minus_epsilon)) / (2 * epsilon)
				print("ND:\n", ND[i][j][k])
	return ND




if __name__=="__main__":
	run(alpha = 0.0001)





