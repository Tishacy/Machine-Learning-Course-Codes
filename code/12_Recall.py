"""
	Recall of a machine learning project.
			 H
	Y	   +	-
	+	  tp 	fn
	-	  fp 	tn

	tp:true positive
	fp:false positive
	tn:true negative
	fn:false negative
"""

import neuralnetwork as NN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# get the dataset for training, cross-validation and test.
def get_dataset():
	np.random.seed(0)
	m = 2000
	X, Y_ = make_moons(m, noise=0.20)
	Y = np.array(list(map(lambda y: [1, 0] if y==1 else [0, 1], Y_)))

	X_train, Y_train = X[:int(0.6*m)], Y[:int(0.6*m)]
	X_cv, Y_cv = X[int(0.6*m):int(0.8*m)], Y[int(0.6*m):int(0.8*m)]
	X_test, Y_test = X[int(0.8*m):], Y[int(0.8*m):]
	return X_train, Y_train, X_cv, Y_cv, X_test, Y_test



# returns the accuracy results
def accuracy(H, Y):
	num = 0
	for i in range(len(H)):
		if H[i][0] == Y[i][0]:
			num += 1
	acry = num / len(H)
	return acry


# returns the recall results
def recall(H, Y):
	# the order of res is the [tp, tn, fp, fn]
	rec = [0, 0, 0, 0]
	for i in range(len(H)):
		if H[i][0] == Y[i][0] and H[i][0] == 1:
			rec[0] += 1
		elif H[i][0] == Y[i][0] and H[i][0] == 0:
			rec[1] += 1
		elif H[i][0] != Y[i][0] and H[i][0] == 1:
			rec[2] += 1
		else:
			rec[3] += 1
	return rec


# get the dataset
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = get_dataset()

# train a neural network with the training dataset
nn = NN.NeuralNetwork(2, 1, 5, 2)
print("Initial parameters of the neural network:\n",nn.ini_Theta)
Theta, steps, Js = nn.gradient_descent(X_train, Y_train, nn.ini_Theta, alpha=0.001)

# get the cross-validation hypotheses values by the parameters trained by neural network
H_cv, A = nn.forward_propagation(X_cv, Theta)
H_cv = np.array([[int(value + 0.5) for row in H_cv for value in row]]).reshape([-1, 2])
ac = accuracy(H_cv, Y_cv)
rec = recall(H_cv, Y_cv)
precision = rec[0] / (rec[0] + rec[2])
recall = rec[0] / (rec[0] + rec[3])
print("""
===== Analysis result =====
	> accuracy: %f,
	> true positive:  %d,
	> true negative:  %d,
	> false positive: %d,
	> false negative: %d,
	> precision: %f,
	> recall: %f
	""" %(ac, rec[0], rec[1], rec[2], rec[3], precision, recall))



