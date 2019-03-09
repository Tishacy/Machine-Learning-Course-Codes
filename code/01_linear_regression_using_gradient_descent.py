"""
	Batch梯度下降法
	2018/8/11
	Tishacy 
"""

import matplotlib.pyplot as plt
import numpy as np
import time


def get_training_dataset(theta_0=1, theta_1=1.5):
    x = np.arange(0, 5, 0.1)
    y = theta_0 * x+ theta_1 * x**2 + 0.3 * np.random.randn(1, len(x))[0]
    # y = theta_0 + theta_1 * np.sin(x) + 0.3*np.random.randn(1, len(x))[0]
    # # change line to curve
    return x, y


def batch_gradient_discent_algorithm(x, y, alpha):
    """
            batch_gradien_discent_algorithm:
                    theta_j := theta_j - alpha * ∂(J(theta_j)) / ∂(theta_j)
                            where 	theta_j is the parameter
                                            alpha is learning rate
                                            J(theta_j) is the cost function

            linear_regression(:
                    hypothese function: h(x) = theta_0 + theta_1 * x
                    cost function: J(theta_0, theta_1) = 1/(2*m) * ∑^m(h(x) - y).^2
                            where 	x is the input
                                            y is the real output
                                            h(x) is the hypothesed output corresponding to x
                                            m is the size of the real dataset

            when put linear regression and batch gradient descent algorithm together,
            the algorithm becomes as follows:
                    theta_0 := theta_0 - alpha * 1/m * ∑^m(theta_0 + theta_1 * x - y)
                    theta_1 := theta_1 - alpha * 1/m * ∑^m((theta_0 + theta_1 * x - y) * x)
            Note: the assignments of theta_0 and theta_1 have to be simultaneous

            actually the algorithm is always as follows, if the cost function is
            mentioned as J:
                    theta_j := theta_j - alpha * 1/m * ∑^m(h(x) - y) * ∂(h(x))/∂(theta_j)
    """
    # initial values of theta_0, theta_1
    theta_0 = 0
    theta_1 = 0
    step = 0
    J = 1 / (2 * len(x)) * sum(np.array(theta_0 *x + theta_1 * x**2 - y)**2)
    # J = 1/(2*len(x)) * sum(np.array(theta_0 + theta_1 * np.sin(x) - y)**2)
    # # change line to curve
    last_J = J

    theta_0s = [theta_0]
    theta_1s = [theta_1]
    steps = [step]
    Js = [J]
    print("正在拟合")
    while (abs(last_J - J) > 1e-6 or step == 0):
        print("step %d  J = %f" % (step, J))
        # gradient descent
        temp_theta_0 = theta_0 - alpha * 1 / \
            len(x) * sum(np.array(theta_0 *x + theta_1 * x**2 - y) * np.array(x))
        temp_theta_1 = theta_1 - alpha * 1 / \
            len(x) * sum(np.array(theta_0 *x + theta_1 * x**2 - y) * np.array(x**2))
        # temp_theta_0 = theta_0 - alpha * 1 / len(x) * sum(np.array(theta_0 + theta_1 * np.sin(x) - y))	# change line to curve
        # temp_theta_1 = theta_1 - alpha * 1 / len(x) * sum(np.array(theta_0 +
        # theta_1 * np.sin(x) - y) * np.sin(np.array(x)))    # change line to
        # curve
        theta_0 = temp_theta_0
        theta_1 = temp_theta_1
        # cost function J
        last_J = J
        J = 1 / (2 * len(x)) * sum(np.array(theta_0 *x+ theta_1 * x**2 - y)**2)
        # J = 1/(2*len(x)) * sum(np.array(theta_0 + theta_1 * np.sin(x) - y)**2)	# change line to curve
        # record result and plot the cost function
        step += 1
        theta_0s.append(theta_0)
        theta_1s.append(theta_1)
        steps.append(step)
        Js.append(J)

    print("""拟合结束
	J = %f
	steps = %d
	theta_0 = %f
	theta_1 = %f""" % (Js[-1], steps[-1], theta_0, theta_1))
    return theta_0s, theta_1s, steps, Js


def fitting_task(real_theta_0=1.5, real_theta_1=1, alpha=0.1):
    x, y = get_training_dataset(real_theta_0, real_theta_1)
    t1 = time.time()
    theta_0s, theta_1s, steps, Js = batch_gradient_discent_algorithm(
        x, y, alpha)
    print("梯度下降共耗时 %.4f 秒" %(time.time()-t1))
    h = theta_0s[-1] *x+ theta_1s[-1] * x**2
    # h = theta_0s[-1] + theta_1s[-1] * np.sin(x)	# change line to curve

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.style.use('bmh')
    plt.figure(figsize=(12, 3.6))
    # plot the training dataset and the fitting line
    ax1 = plt.subplot(131)
    plt.plot(x, y, '.', color='r', alpha=0.8, label="training dataset")
    plt.plot(x, h, '-', color='k', alpha=0.8, label="fitting line")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plot the decreasing of J
    ax2 = plt.subplot(132)
    plt.plot(steps, Js, '-', color='r', alpha=0.7, label="cost function")
    plt.scatter(steps, Js, marker='o', s=5, c='k')
    plt.xlabel(r'No. of iterations')
    plt.ylabel(r'cost function')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    # plot the trajectory of theta_0, theta_1
    ax3 = plt.subplot(133)
    plt.scatter(theta_0s[0], theta_1s[0], s=25, c='b', label="start_point")
    plt.scatter(theta_0s[-1], theta_1s[-1], s=25, c='r', label="end_point")
    plt.scatter(real_theta_0, real_theta_1, s=25, c='k', label="real_point")
    plt.plot(theta_0s, theta_1s, '-', color='k', alpha=0.5,
             label=r"trajectory of $\theta_0$, $\theta_1$")
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fitting_task(1, 1.5, 0.0001)
