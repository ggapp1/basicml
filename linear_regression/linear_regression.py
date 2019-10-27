import numpy as np
import pandas as pd

def load_dataset():
	"""
	Load the housing prices dataset

	Return:
	train_set_x, train_set_y -- (x,y) of training set
	test_set_x, test_set_y -- (x,y) of testing set
	"""	
	with open('../datasets/housing_prices.csv', "r") as dataset:
		dataset = pd.read_csv(dataset)
		dataset_x_orig = np.array(dataset.select_dtypes(include=[np.number]))
		dataset_y_orig = np.array(dataset['SalePrice'])

	#divide train and test set with aprox 80/20 proportion
	train_set_x = dataset_x_orig[:1260]
	train_set_y = dataset_y_orig[:1260]
	test_set_x = dataset_x_orig[1260:] 
	test_set_y = dataset_y_orig[1260:]

	return train_set_x, train_set_y.reshape((1, train_set_y.shape[0])), test_set_x, test_set_y.reshape((1, test_set_y.shape[0]))

def cost_function(Y, A):
	"""
	Compute the loss function (square difference)

	Arguments:
	Y -- A scalar or numpy array represeting the actual value.
	A -- A scalar or numpy array representing the predicted value.
	Return:
	cost -- cost for given parameters
	"""
	return (np.pow(Y - A))

def gradient_descent(X, Y, w, b):
	"""
	Implement the gradient of the cost function for propagation

	Arguments:
	w -- weights, a numpy array of size (number of features, 1)
	b -- bias, a scalar
	X -- data of size (number of features, number of examples)
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

	Return:
	dw -- gradient of the loss with respect to w, thus same shape as w
	db -- gradient of the loss with respect to b, thus same shape as b
	cost -- negative log-likelihood cost for logistic regression
	"""
	A = np.dot(w.T, X) + b	
	#number of features
	m = X.shape[0]
#	dz = A - Y
#	db = 1/m * np.sum(dz)
#	dw = 1/m * np.dot(X, dz.T) 

	return dw, db, cost_function(Y, A)

def optimize(X, Y, w, b, epochs, learning_rate):
	"""
	This function optimizes w and b by running a gradient descent algorithm
	
	Arguments:
	w -- weights, a numpy array of size (number of features, 1)
	b -- bias, a scalar
	X -- data of shape (number of features, number of examples)
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
	epochs -- number of iterations of the optimization loop
	learning_rate -- learning rate of the gradient descent update rule
	
	Returns:
	w, b  -- weights w and bias b
	dw, db --gradients of the weights and bias with respect to the cost function
	costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
	"""
	costs = []
	print('\nTraining... \n')
	for i in range(epochs):		 							
		dw, db, cost = gradient_descent(X, Y, w, b)
		costs.append(cost)
		if(i%200 == 0 ):
			print("cost after {0}th iteration: {1}".format(i, cost))
		w = w - learning_rate * dw
		b = b - learning_rate * db

	return w, b, dw, db, costs


def predict(X, w, b):
	'''
	Predict whether the label is 0 or 1 using learned parameters (w, b)
	
	Arguments:
	X -- data of size (number of features, number of examples)
	w -- weights, a numpy array of size (number of features, 1)
	b -- bias, a scalar

	Returns:
	A -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	'''
	m = X.shape[1]
	w = w.reshape(X.shape[0], 1)
	A = np.dot(w.T, X) + b

	return A	

#def normalEquation(X, Y, theta):	

def linear_regression(X_train, Y_train, X_test, Y_test, epochs = 2000, learning_rate = 0.5):
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
		theta = new_theta

	return theta	


def computeError(X, Y, theta):

	error = (np.sum(computeL(computeH(X, theta),Y)))
	return error

def main():
	train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()
	print(train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)


if __name__ == '__main__':
	main()