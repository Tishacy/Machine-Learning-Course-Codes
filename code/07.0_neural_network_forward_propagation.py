"""
	Forward propagation
	2018/8/13
	Tishacy
"""

import numpy as np

# 定义输入数据 X
SEED = 23455
rdm = np.random.RandomState(SEED)
X = rdm.rand(5, 4)

# 创建神经网络，不考虑bias节点
# 共3层，输入层 4 个节点，隐含层 5 个节点，输出层 3 个节点
theta_0 = np.ones((4, 5))
theta_1 = np.ones((5, 3))
Theta = [theta_0, theta_1]

# 定义sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义前向传播过程
def froward_propagation_no_bias(X, Theta):
    a = X
    for theta in Theta:
        a = sigmoid(np.dot(a, theta))
    h = a
    return h

# 输出前向传播结果
h = froward_propagation_no_bias(X, Theta)
print(h)

"""
[[0.98842249 0.98842249 0.98842249]
 [0.98574803 0.98574803 0.98574803]
 [0.98608855 0.98608855 0.98608855]
 [0.99054895 0.99054895 0.99054895]
 [0.98298043 0.98298043 0.98298043]]
"""
