"""
	正则化
	2018/8/13
	Tishacy
	
	//// 无正则化->正则化所修改的部分已用[]标出 ////
	
	对于 linear regression (回归)问题：
		h = theta^T * X
		（1）无正则化：
			Cost function J:
				J = 1 / (2*m) * [∑^m(h-y).^2]
			Gradient descent algorithm:
				theta_j := [theta_j] - alpha * (1/m) * ∑^m((h-y) .* x)
		（2）正则化：
			Cost function J:
				J = 1 / (2*m) * [∑^m(h-y).^2 + ∑^n(theta.^2)]
			Gradient descent algorithm:
				theta_0 := [theta_0] - alpha * (1/m) * ∑^m((h-y) .* x)
				theta_j := [theta_j * (1 - alpha * lambda / m)] - alpha * (1/m) * ∑^m((h-y) .* x)  	when j>=1

	对于 logistics regression (分类)问题：
		h = 1 / (1 + exp(-theta^T * X))
		(1) 无正则化：
			Cost function J:
				J = -(1/m) * ∑^m(y*ln(h) + (1-y)*ln(1-h))
			Gradient function algorithem:
				theta_j := [theta_j] - alpha * (1/m) * ∑^m((h-y) .* x)
		(2) 正则化：
			Cost function J:
				J = -(1/m) * ∑^m(y*ln(h) + (1-y)*ln(1-h)) + 1/(2*m) * [∑^n(theta.^2)]
			Gradient descent algorithm:
				theta_0 := [theta_0] - alpha * (1/m) * ∑^m((h-y) .* x)
				theta_j := [theta_j * (1 - alpha * lambda / m)] - alpha * (1/m) * ∑^m((h-y) .* x)  	when j>=1

	注意：一般不将theta_0正则化，而是从theta_1一直到theta_n，因此在进行梯度下降算法时，应将theta_0单独拿出来
"""
