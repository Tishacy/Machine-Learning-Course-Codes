"""
	带bias的神经网络前向传播
	2018/8/13
	Tishacy

	注意：在这里，神经网络向量化（矩阵化）的方式统一如下：

		(1) 所有的节点在矩阵中都横着排(每个节点数据占1列)，即：
				o node_0
				o node_1  ->  [node_0, node_1, node_2]
				o node_2
			同样，如果某一层a增加一个节点，就相当于在该层的a矩阵中
			增加一列

		(2) 从第i层到第i+1层的权重weight(即theta)矩阵的维度为：
				如果不考虑bias节点：
					dim(θ) = num_nodes(i) × num_nodes(i+1)
				如果考虑bias节点：
					dim(θ) = num_nodes(i) + 1 × num_nodes(i+1)
			其中，num_nodes(i)表示第i层的节点数，计算num_nodes时
				不算bias节点。

			例如：
					  o				不考虑bias节点：
			  o   	  o 				dim(θ) = 3 × 5
			  o   	  o  	==>   	考虑bias节点：
			  o   	  o 				dim(θ) = (3+1) × 5
					  o 				   	   = 4 × 5
			第1层 θ	第2层
"""

import numpy as np

# 定义输入数据 X
SEED = 23455
rdm = np.random.RandomState(SEED)
X = rdm.rand(5, 4)

# 创建神经网络
# 共3层，输入层 4 个节点，第二层 5 个节点， 输出层 3 个节点 (不含bias节点)
theta_0 = np.ones([4+1, 5])
theta_1 = np.ones([5+1, 3])
Theta = [theta_0, theta_1]

# 定义sigmoid激活函数sigmoid
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# 给每一层增加一个bias节点
def add_bias(a_nodes, bias_value):
	m = len(a_nodes)
	bias_node = bias_value * np.ones([m, 1])	# 定义一个bias节点
	return np.c_[bias_node, a_nodes]

# 定义前向传播过程
def forward_propagation_with_bias(X, Theta, bias_value=0.1):
	a = X
	for theta in Theta:
		a = add_bias(a, bias_value)
		a = sigmoid(np.dot(a, theta))
	h = a
	return h

# 输出前向传播结果
h = forward_propagation_with_bias(X, Theta, 1)
print(h)

"""
[[0.99692475 0.99692475 0.99692475]
 [0.99662883 0.99662883 0.99662883]
 [0.99666564 0.99666564 0.99666564]
 [0.99717422 0.99717422 0.99717422]
 [0.99633687 0.99633687 0.99633687]]
"""