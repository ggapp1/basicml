import numpy as np

#hypothesis
def computeH(X, theta):
	return (np.matmul(theta.transpose(), X))

#loss
def computeL(h, Y):
	return np.subtract(h, Y)

def gradientDescent(X, Y, theta):
	h = computeH(X, theta)
	l = computeL(h, Y)
	return(np.matmul(l, X.transpose())[0])
	

def multivariableRegression(X, Y, theta):
	#learnig rate
	alpha = 0.05
	#number of features
	n = len(theta)
	#batch size
	m = 3.0 

	new_theta = theta

	for i in range(2000):
		gradients = gradientDescent(X, Y, theta)		
		for i in range(n):
			new_theta[i][0] =  theta[i] - (alpha * 1/m * gradients[i])
		new_theta = theta

	return theta	


def computeError(X, Y, theta):

	error = (np.sum(computeL(computeH(X, theta),Y)))
	return error

def main():
	Y = np.array([1.0, 4.0, 6.0])

	X = np.array([[1.0,1.0,1.0], [0.0, 1.0, 2.0], [0.0, 1.0, 1.0]])

	theta = np.array([[42.0], [10.0], [12.0]])	

	new_theta = singlevariableRegression(X, Y, theta)
	print("final error:")
	print(computeError(X, Y, new_theta))
	print("final theta:")
	print(new_theta)

if __name__ == '__main__':
	main()