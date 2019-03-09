"""
	A test for Neural Network Module of regularization

	similar to /Tesorflow/regularization.py
"""

import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt


# 生成数据集
seed = 22
m = 500
rdm = np.random.RandomState(seed)
X = rdm.randn(m, 2)
Y = np.array([[int(x0**2 + x1**2 < 2)] for (x0, x1) in X])
Y[0] = [1]
Y[1] = [0]
Y_c = ['red' if y[0] else 'blue' for y in Y]
print("X:\n", X)
print("Y:\n", Y)

# 搭建神经网络
NN = nn.NeuralNetwork(2, 1, 4, 1)


# 用无正则化、正则化两种情况训练神经网络
Theta_nR, steps_nR, Js_nR = NN.gradient_descent(X, Y, NN.ini_Theta, alpha=0.002, lamda=0)
Theta_R, steps_R, Js_R = NN.gradient_descent(X, Y, NN.ini_Theta, alpha=0.002, lamda=1)


# 定义测试
xx0, xx1 = np.meshgrid(np.arange(-3, 3, 0.01), np.arange(-3, 3, 0.01))
X0_test = xx0.ravel()
X1_test = xx1.ravel()
h_nR = NN.forward_propagation(np.c_[X0_test, X1_test], Theta_nR)[0]
h_nR = h_nR.reshape(xx0.shape)
h_R = NN.forward_propagation(np.c_[X0_test, X1_test], Theta_R)[0]
h_R = h_R.reshape(xx0.shape)


# 画出训练数据及决策边界
plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.contour(xx0, xx1, h_nR, levels=[0.5], colors='k', alpha=0.7)
plt.contourf(xx0, xx1, h_nR, 1, cmap="RdBu_r")
plt.scatter(X[:, 0], X[:, 1], c=Y_c, s=18, alpha=0.8)
plt.title('without regularization')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.subplot(122)
plt.contour(xx0, xx1, h_R, levels=[0.5], colors='k', alpha=0.7)
plt.contourf(xx0, xx1, h_R, 1, cmap="RdBu_r")
plt.scatter(X[:, 0], X[:, 1], c=Y_c, s=18, alpha=0.9)
plt.title('regularization')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.tight_layout()
plt.show()
