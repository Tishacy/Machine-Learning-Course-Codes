"""
	Confirming the forward propagation
	通过使用TensorFlow来构造与7.0相同的神经网络结构，
	并喂入相同的数据，使用相同的激活函数，相同的bias来
	检测7.0中的数学计算是否正确。
	2018/8/13
	Tishacy
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


# 创建神经网络，与7.0相同
x = tf.placeholder(tf.float32, shape=(None, 4))
h = tf.placeholder(tf.float32, shape=(None, 3))
theta_0 = tf.Variable(tf.ones([4, 5]))
theta_1 = tf.Variable(tf.ones([5, 3]))

# 定义输入函数 X
SEED = 23455
rdm = np.random.RandomState(SEED)
X = rdm.rand(5, 4)

# 定义前向传播过程
a = tf.nn.sigmoid(tf.matmul(x, theta_0))
y = tf.nn.sigmoid(tf.matmul(a, theta_1))

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(y, feed_dict={x: X}))
	
"""
[[0.98842245 0.98842245 0.98842245]
 [0.985748   0.985748   0.985748  ]
 [0.9860886  0.9860886  0.9860886 ]
 [0.9905489  0.9905489  0.9905489 ]
 [0.9829805  0.9829805  0.9829805 ]]
 与7.0中的结果大致相同，但是精度不同。
"""


