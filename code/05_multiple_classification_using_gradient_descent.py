"""
	多分类问题
"""

import matplotlib.pyplot as plt
import numpy as np



def get_dataset():
	n = 2	# n features
	m = 400	# m samples 
	X = np.random.uniform(0.0, 4.0, (n, m))
	num_class = 4	# 分类数
	# 将下面区域划分为4个
	y = np.ones(m)
	y = np.array([list(map(lambda x_0, x_1: 1 if x_0*x_1 < 2 and x_1 < 2 
						 else (2 if 2 <= x_0 and x_1 < 2 
						   else (3 if x_0 < 2 and 2 <= x_1 
						   	 else 4)), X[0], X[1]))])
	return X, y, num_class


def logistic_regression(X, y, num_class, alpha=0.01):
	"""
	将原始训练数据 X 与标记集 y 进行预处理
		X:  增加一行值为1的数据
	 	Y： 将每个单类设为1 其余类设为0 例如：将 y==1 与 y!=1 分成两类，分别标记为1，0
		因此会产生num_class组训练数据，将原本的y数据(1 × m)扩展成为Y数据(num_class × m)
		其中每一行代表一组训练标记集， 如：
			当有4类时，可将y扩展为Y:
			y = [[1 1 1 2 2 2 3 3 3 4 4 ]]
			Y = [[1 1 1 0 0 0 0 0 0 0 0 0],
				 [0 0 0 1 1 1 0 0 0 0 0 0],
				 [0 0 0 0 0 0 1 1 1 0 0 0],
				 [0 0 0 0 0 0 0 0 0 1 1 1]]
	"""
	m = len(y[0])
	X = np.r_[np.ones((1, len(X[0]))), X]
	n = len(X)
	Y = np.ones((num_class, m))
	for i in range(len(Y)):
		Y[i] = list(map(lambda x: 1 if x==i+1 else 0, y[0]))
	print(X, Y)
	"""
		Hypothesis function:
			H = 1 / (1 + exp(-theta^T * X))		(num_class × m)
			where 	theta is the matrix of parameters  (n × num_class)
					X is the training sample (n × m)
		
		Cost funciton:
			J = - 1/m * ∑^m(y.*ln(h) + (1-y).*ln(1-h))  (num_class × 1)
		
		Gradient descent algorithm:
			theta_j := theta_j - alpha * 1/m ∑^m((H - y).*x)
			which is:
				theta = theta - alpha * 1/m(X * (H - Y)^T)
	"""
	theta = np.zeros((n, num_class))
	H = 1 / (1 + np.exp(np.dot(-np.transpose(theta), X)))
	J = - 1/m * np.sum((Y*np.log(H) + (1-Y)*np.log(1-H)))	# J is always a value
	#J = - 1/m * np.transpose([np.sum(Y*np.log(H) + (1-Y)*np.log(1-H), axis=1)])
	last_J = J
	step = 0
	Js = [J]
	#Js = [np.transpose(J)[0]]
	steps = [step]
	print("正在拟合")
	while True:
		print("step: %d  J: %.6f" %(step, J))
		#print("step:%d  J:" %(step), np.transpose(J)[0])
		last_J = J
		theta = theta - alpha * 1 / m * np.dot(X, np.transpose(H - Y))
		H = 1 / (1 + np.exp(np.dot(-np.transpose(theta), X)))
		J = - 1/m * np.sum((Y*np.log(H) + (1-Y)*np.log(1-H)))
		#J = - 1/m * np.transpose([np.sum(Y*np.log(H) + (1-Y)*np.log(1-H), axis=1)])
		step += 1
		Js.append(J)
		#Js.append(np.transpose(J)[0])
		steps.append(step)
		if last_J - J < 1e-5:
		#if (last_J - J < 1e-5).all():
			break
	print("拟合结束")
	print("J = %.6f" %J)
	#print("J:\n", np.transpose(J)[0])
	print("steps = %d" %step)
	print("theta:\n", theta)
	return theta, np.array(steps), np.array(Js)



def train_model(alpha = 1):
	# 获取数据并训练模型
	X, y, num_class = get_dataset()
	theta, steps, Js = logistic_regression(X, y, num_class, alpha)

	colors = list(map(lambda x: 'indianred' if x==1 else ('steelblue' if x==2 else ('dimgrey' if x==3 else 'orchid')), y[0]))
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.figure(figsize=(9, 4))
	
	# 画出training dataset 和拟合曲线b
	plt.subplot(121)
	plt.scatter(X[0], X[1], color=colors, s=18, label='trained data')
	# 计算boundary line
	colors = ['indianred', 'steelblue', 'dimgrey', 'orchid']
	x = np.arange(0, 4, 0.01)
	boundary_lines = np.ones((len(y[0]), num_class))
	for i in range(num_class):
		boundary_lines[:, i] = (-theta[:, i][0] - theta[:, i][1] * x) / theta[:, i][2]
		plt.plot(x, boundary_lines[:, i], '--', alpha = 0.7, color=colors[i])
	plt.xlabel(r"$feature_0$", fontsize=12)
	plt.ylabel(r"$feature_1$", fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend(loc='upper right')

	# 画出cost function J的下降曲线
	plt.subplot(122)
	plt.plot(steps, Js, color='indianred', alpha=0.7, label=r"cost funciton $J$")
	#for i, J in enumerate(np.transpose(Js)):
	#	plt.plot(steps, J, color=colors[i], alpha=0.7, label=r"$feature_%d$" %(i+1))
	plt.xlabel('No. of iterations', fontsize=12)
	plt.ylabel(r'cost function $J$', fontsize=12)
	plt.legend()
	#plt.show()
	plt.suptitle("Performance of Training the Model", fontsize=15)
	return theta


def multi_classify(feature_0, featrue_1, theta):
	# 给定数据计算结果
	H = 1 / (1 + np.exp(np.dot(-np.transpose(theta), np.array([[1],[feature_0],[featrue_1]]))))
	label = np.where(H == max(H))[0][0]+1
	return label


def test_set(alpha=1):
	# 原训练数据集
	X, y, num_class = get_dataset()
	# 生成m组数据作为测试集 X_
	m = 1000
	X_ = np.random.normal(2.0, 1.0, (2, m))
	# 训练模型
	theta = train_model(alpha)
	# 获取结果数据
	labels = [multi_classify(X_[0][i], X_[1][i], theta) for i in range(m)]
	# 将结果可视化
	plt.figure(figsize=(6, 5))
	colors_ = list(map(lambda x: 'indianred' if x==1 else ('steelblue' if x==2 else ('dimgrey' if x==3 else 'orchid')), labels))
	colors = list(map(lambda x: 'indianred' if x==1 else ('steelblue' if x==2 else ('dimgrey' if x==3 else 'orchid')), y[0]))
	plt.scatter(X_[0], X_[1], color=colors_, s=18, label='test data')
	plt.scatter(X[0], X[1], color=colors, s=18, label='trained data', alpha=0.3)
	plt.xlabel(r"$feature_0$", fontsize=12)
	plt.ylabel(r"$feature_1$", fontsize=12)
	plt.xlim(0, 4)
	plt.ylim(0, 4)
	plt.legend(loc='upper right')
	plt.title("Tests of the Model", fontsize=15)

	plt.show()


if __name__=="__main__":
	#theta = train_model()
	#label = multi_classify(1.7, 2.5, theta)
	#print("the class of given features is %d" %label)
	test_set(0.8)


