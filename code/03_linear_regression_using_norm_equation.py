"""
	正规方程求解多元线性回归
	2018/8/11
	Tishacy

		正则方程：
		theta = pinv(X^T * X) * X^T * y
"""

import numpy as np


def get_training_dataset():
    n = 10  # 一共有10个feature
    m = 50  # 每个feature有50个样本点
    # x: 50 × 10
    x = np.random.rand(m, n)  # 已经得到了normalized数据
    # theta: 10 × 1
    theta = np.transpose(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    # y: 50 × 1
    y = np.dot(x, theta) + 0.3 * np.random.rand(m, 1)
    #print(x.shape, theta.shape, y.shape)
    return x, y, theta


def multivariate_linear_regression_norm(x, y):
    theta = np.dot(np.dot(np.linalg.pinv(
        np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return theta


def fitting_task():
    x, y, real_theta = get_training_dataset()
    theta = multivariate_linear_regression_norm(x, y)
    std_err = sum((theta - real_theta)**2) / len(theta)
    print("theta:\n", np.transpose(theta)[0])
    print("standard error: %.6f" % std_err)


if __name__ == "__main__":
    fitting_task()
