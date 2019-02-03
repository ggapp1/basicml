import numpy as np


def computeH(m, theta, x, y):

	return (np.matmul(theta.transpose(), x) - y)

def gradientDescent(alpha, theta, X, y, m, n):
	new_theta = [0,0]
	for i in range(0,n):
		new_theta[i] = 1/m - (alpha * computeH(m,theta,X, y)) 
	return new_theta



def singlevariableRegression(alpha, X, y, m):
	theta = 1

def multivariableRegression(alpha, X, y, m, n):
	theta = 1



def main():
	alpha = 1
	m = 1
	theta = np.array([[3], [3]])
	x = np.array([[1], [2]])
	y = np.array([[1], [10]])
	n = 2
	print(gradientDescent(alpha, theta, x, y, m, n))



if __name__ == '__main__':
	main()