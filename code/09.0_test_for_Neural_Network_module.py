"""
	A test for Neural Network Module
"""
from sklearn.datasets import make_moons
import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt



# 画出决策边界（decision boundary）
def plot_decision_boudary(X, Y, NN, Theta):
	x_0_min, x_0_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	x_1_min, x_1_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx_0, xx_1 = np.meshgrid(np.arange(x_0_min, x_0_max, 0.01), np.arange(x_1_min, x_1_max, 0.01))

	Z, A = NN.forward_propagation(np.c_[xx_0.ravel(), xx_1.ravel()], Theta)
	#Z = np.array([y[0] if (np.where(y==np.max(y))[0][0]==0) else y[1]  for y in Z])
	Z = np.array([y[0] for y in Z])
	Z = Z.reshape(xx_0.shape)
	
	c_ = [1 if (y == [1, 0]).all() else 0  for y in Y]
	plt.contour(xx_0, xx_1, Z, 1, linestyles='dashed', linewidths=0.5, colors='k')
	plt.contourf(xx_0, xx_1, Z, 1, cmap="RdBu_r")
	plt.scatter(X_train[:,0], X_train[:,1], s=18, c=c_, cmap="RdBu_r", label='train set')
	plt.legend(framealpha=0.5)


# 获取数据集并分为训练集，交叉验证集，测试集
np.random.seed(0)
m = 2000
X, Y_ = make_moons(m, noise=0.20)
Y = np.array(list(map(lambda y: [1, 0] if y==1 else [0, 1], Y_)))

X_train, Y_train = X[:int(0.6*m)], Y[:int(0.6*m)]
X_cv, Y_cv = X[int(0.6*m):int(0.8*m)], Y[int(0.6*m):int(0.8*m)]
X_test, Y_test = X[int(0.8*m):], Y[int(0.8*m):]


# 搭建神经网络
NN = nn.NeuralNetwork (2, 4, 4, 2)

# 训练模型
Theta, steps, Js = NN.gradient_descent(X_train, Y_train, NN.ini_Theta, alpha=0.001, threshold=1e-7)

# 获得训练后的数据进行交叉验证与测试
h_cv, A = NN.forward_propagation(X_cv, Theta)
J_cv = NN.cost_function(Y_cv, h_cv, Theta)
h_test, A = NN.forward_propagation(X_test, Theta)
J_test = NN.cost_function(Y_test, h_test, Theta)

print("===== Fitting Result of Cross Validation =====")
print("    J_cv = %.6f" %J_cv)
print("    J_test = %.6f" %J_test)


plt.figure(figsize=(10, 4))
# 画出图像
plt.subplot(121)
c_train = [1 if (y == [1, 0]).all() else 0  for y in Y_train]
c_cv = [1 if (np.where(y==np.max(y))[0][0]==0) else 0  for y in h_cv]

plt.scatter(X_train[:,0], X_train[:,1], s=18, c=c_train, alpha=0.5, cmap="RdBu_r", label="train set")
plt.scatter(X_cv[:,0], X_cv[:,1], s=18, c=c_cv, cmap="RdBu_r", label="cross validation set")
plt.legend()

# 画出决策边界
plt.subplot(122)
plot_decision_boudary(X_train, Y_train, NN, Theta)



# # 画出学习曲线
# scale_hidden_layers = np.arange(1, 5)
# scale_lamda = np.arange(0, 0.05, 0.005)
# num_slices = 10
# nn.learning_curve(NN, scale_hidden_layers, scale_lamda, num_slices, X_train, Y_train, alpha=0.001)

plt.show()


