"""
	多元线性回归（梯度下降法） 
        另一版本（样本x的行列互换）
	2018/8/11
	Tishacy
"""

import numpy as np
import matplotlib.pyplot as plt

"""
	Hypothesis function:	h(x) = x * theta
		where	theta is the vector of n+1 parameters of theta_j
					[theta_0, theta_1, theta_2 ... theta_n]^T
                T means the transpose of a vector or a matrix
				x is the matrix of n+1 features of x_i
					[x_0, x_1, x_2 ... x_n]
                    
                            feature 0   feature 1   feature 2   ...   feature n
                              x_0[0]      x_1[0]      x_2[0]    ...    x_3[0]
                    samples   x_0[1]      x_1[1]      x_2[1]    ...    x_3[1]
                            ...................................................
                              x_0[m]      x_1[m]      x_2[m]    ...    x_3[m]

					Note:  (1) x_0 is commonly equals to 1
                           (2) x'dimension is m × n
				
	Cost function:	J(theta) = 1/(2*m) * ∑^m(h(x) - y).^2

	Gradient descent algorithm:	theta_j := theta_j - alpha * ∂(J(theta_j)) / ∂(theta_j)
		In multivariate linear regression, the algorithm is as follows:
			theta_j := theta_j - alpha * 1/m * np.dot(x^T, (h(x) - y))

	Feature scaling or normalization (to make sure the converge be faster):
		each feature x_i is normalized as:  norm_x_i := (x_i - miu_i)/(max(x_i) - min(x_i))
"""

def get_training_dataset():
    n = 10  # 一共有10个feature
    m = 50  # 每个feature有50个样本点
    # x: 50 × 10
    x = np.random.rand(m, n)  # 已经得到了normalized数据
    # theta: 10 × 1
    theta = np.transpose(np.array([[1, 2, 3, 4, 5.5, 6, 7, 8, 9, 10]]))
    # y: 50 × 1
    y = np.dot(x, theta) + 0.3 * np.random.rand(m, 1)
    #print(x.shape, theta.shape, y.shape)
    return x, y, theta


def multivariate_linear_regression_gradient_descent(x, y, alpha):
    # 初始化模型
    m = len(x[0])
    theta = np.zeros([10, 1])  # 10 × 1
    h = np.dot(x, theta)  # 50 × 1
    J = 1 / (2 * m) * sum(sum((h - y)**2))  # 1 × 1

    last_J = J
    step = 0
    Js = [J]
    steps = [step]
    print("正在拟合")
    while abs(last_J - J) > 1e-5 or step == 0:
        print("step %d  J = %f" % (step, J))
        print(np.dot(np.transpose(x), (h - y)).shape)
        theta = theta - alpha * 1 / m * np.dot(np.transpose(x), (h - y))
        last_J = J
        h = np.dot(x, theta)
        J = 1 / (2 * m) * sum(sum((h - y)**2))
        step += 1
        Js.append(J)
        steps.append(step)
    print("""拟合结束\n    J = %f\n    steps = %d\n    theta:\n    """ %
          (J, step), np.transpose(theta)[0])
    return theta, steps, Js


def fitting_task(alpha=0.1):
    x, y, real_theta = get_training_dataset()
    theta, steps, Js = multivariate_linear_regression_gradient_descent(
        x, y, alpha)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.style.use('bmh')
    ax = plt.figure(figsize=(9, 4))

    # 画出cost function J的下降曲线
    plt.subplot(121)
    plt.plot(steps, Js, '-', color='r', alpha=0.5,
             label=r"cost function $J$ decay")
    plt.xlabel(r'No. of iterations', fontsize=12)
    plt.ylabel(r'cost function $J$', fontsize=12)
    plt.title(r'Decay of $J$')
    plt.legend()

    # 画出theta对比图
    plt.subplot(122)
    plt.plot(theta, real_theta, 'o', color='r', label=r"$\theta$ values")
    x = y = np.arange(min(theta), max(theta) + 1)
    std_err = sum((theta - real_theta)**2) / len(theta)
    plt.plot(x, y, '--', color='k', alpha=0.5, label=r"comparison line")
    plt.xlabel(r"$\theta$s of fitting results", fontsize=12)
    plt.ylabel(r"real $\theta$s", fontsize=12)
    plt.title(r"standard error = %.6f" % (std_err))
    plt.legend()

    plt.suptitle(r"""Multivariate Linear Regression using
		Gradient Descent Algorithm""", fontsize=15)
    ax.subplots_adjust(bottom=0.1, top=0.8)
    plt.show()


if __name__ == "__main__":
    fitting_task(alpha=0.1)
