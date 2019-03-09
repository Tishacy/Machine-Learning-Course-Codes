"""
	Stochastic Gradient Descent Algorithm

	If you have a big dataset, for example 300,000,000, the batch gradient desenct would be computationally expensive for summing the differences of all data while computing the cost function J and its derivatives as follows in each iteration.

		J = 1/(2*m) * ∑(h-y).^2
		
	In big data, people usually use stochastic gradient descent algorithm instead of the batch to avoid this. In stochastic gradient descent, we don't sum the differences of all data just to compute the J and update the paramters in one iteration, but to compute just the diffrence of one pair of data to update the parameters.
	
	The algorithm is as follows:

		1. Randomly shuffle the dataset
		2. for i in N times
			{
				for data in dataset
					{
						theta_j := theta_j - alpha * ∂J/∂(theta_j)
					}
			}
			i = 0,1,2...N
			N = 1 ~ 5, depends on the size of the dataset.
"""
