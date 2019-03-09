"""
	Logistic回归————二元分类问题
	2018/8/12
	Tishacy
"""
import math
import numpy as np
import matplotlib.pyplot as plt


"""
	Hypothesis:
		h(x) = 1 / (1 + exp(-theta^T * X))
			where	h(x) is the hypothesis function, 1 × m
					theta is the vector of parameters, n × 1
					X is the matrix of inputs with n features and m samples, n × m
					T means the transpose of a vector or a matrix 

	Cost function J:
		J(theta) = 1/m ∑^m(δ) = -1/m ∑^m(y.*ln(h(x)) + (1-y)).*ln(1-h(x))
			where	m is the size of sample data
					y is the vector of labels already known, 1 × m
						Note that y can only be 0 or 1 in (binary) classification problem
					δ = -(y.*ln(h(x)) + (1-y)).*ln(1-h(x)) could be more easily understood by follows:
						if y = 0,  	δ = -ln(1-h(x))
							when h(x) -> 0 = y, δ -> 0, which means the cost is smaller for h(x) -> y
							when h(x) -> 1, δ -> +∞, which means the cost is extremely huge for h(x) staying away from y
						if y = 1, 	δ = -ln(h(x))
							when h(x) -> 0, δ -> +∞, which means the cost is extremely huge for h(x) staying away from y
							when h(x) -> 1 = y, δ -> 0, which means the cost is smaller for h(x) -> y
						In all, δ is meant to quantify the cost for difference between hypothesis h(x) and labels y
	
	Gradient descent algorithm:
		theta_j := theta_j - alpha * (1/m) * ∑^m((h(x) - y) .* x)
			where	alpha is the learning rate
"""


def get_dataset():
	# X: n × m
	n = 3
	m = 400
	X = np.random.randn(n, m)
	X[0] = np.ones(np.size(X[0]))
	X[1] = X[1]**2
	X[2] = X[2]**2

	# theta: n × 1
	theta = np.transpose([[-2, 0.8, 2.1]])
	# y : 1 × m
	y = np.array([list(map(			# if (-theta^T * X) > 0, y=1, otherwise y=0
		lambda x: 1 if (x > 0 and np.random.rand() < 0.95) or (x < 0 and np.random.rand() < 0.05) else 0,
		np.dot(np.transpose(theta), X)[0]))])
	return X, y, theta


def logistic_regression(X, y, alpha=0.01):
	# initial settings
	m = len(y)
	theta = np.transpose([np.zeros(3)])  # n × 1

	h = 1 / (1 + np.exp(-np.dot(np.transpose(theta), X)))  # 1 × m
	J = -1 / m * sum(sum(y * np.log(h) + (1 - y) * np.log(1 - h)))		# 1 × 1
	last_J = J
	step = 0
	Js = [J]
	steps = [step]
	#print('h:\n', h, '\nJ:\n', J)
	print("正在拟合")
	while True:
		print("step = %d, J = %.6f" % (step, J))
		# updating
		theta = theta - alpha * (1 / m) * np.dot(X, np.transpose(h - y))
		last_J = J
		h = 1 / (1 + np.exp(-np.dot(np.transpose(theta), X)))
		J = -1 / m * sum(sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
		step += 1
		# record the data
		Js.append(J)
		steps.append(step)
		if abs(last_J - J) < 1e-2 or math.isnan(J):
			break

	print("""拟合结束\n    J = %.4f\n    steps = %d\n    theta:\n    """ %
		  (Js[-1], step), np.transpose(theta)[0])
	return theta, steps, Js


def fitting_task(alpha=0.1):
	X, y, real_theta = get_dataset()
	theta, steps, Js = logistic_regression(X, y, alpha)

	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.style.use('bmh')
	ax = plt.figure(figsize=(9, 4))

	# 画出training dataset 和拟合曲线b
	plt.subplot(121)

	colors = list(map(lambda x: 'r' if x > 0 else 'b', y[0]))
	plt.scatter(np.sqrt(X[1]), np.sqrt(X[2]), c=colors, s=20, alpha=0.5, label='taining dataset')
	# 计算decision boundary
	x_1 = np.arange(0, -theta[0][0], 0.01)
	x_2 = np.sqrt((-theta[0][0]-theta[1][0] * x_1**2)/theta[2][0])
	plt.plot(x_1, x_2, '--', color='k', alpha=0.7, label='decision boundary')
	plt.xlabel(r'$X_1$', fontsize=12)
	plt.ylabel(r'$X_2$', fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend()

	# 画出cost function J的下降曲线
	plt.subplot(122)
	plt.plot(steps, Js, '-', color='r', alpha=0.5,
			 label=r"cost function $J$ decay")
	plt.xlabel(r'No. of iterations', fontsize=12)
	plt.ylabel(r'cost function $J$', fontsize=12)
	plt.legend()

	plt.show()


if __name__ == "__main__":
	fitting_task(0.001)
