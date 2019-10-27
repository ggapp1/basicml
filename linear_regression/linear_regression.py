import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def load_dataset():
	"""
	Load the boston housing prices dataset

	Return:
	train_set_x, train_set_y -- (x,y) of training set
	test_set_x, test_set_y -- (x,y) of testing set
	"""	
	dataset = load_boston()
	boston = pd.DataFrame(dataset.data, columns=dataset.feature_names)
	boston['MEDV'] = dataset.target
	#divide train and test set with aprox 80/20 proportion
	X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
	Y = boston['MEDV']
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
	return np.array(X_train).T, np.array(Y_train).reshape((1,Y_train.shape[0] )), np.array(X_test).T, np.array(Y_test).T.reshape((1,Y_test.shape[0]))

def cost_function(Y, A):
	"""
	Compute the loss function (square difference)

	Arguments:
	Y -- A scalar or numpy array represeting the actual value.
	A -- A scalar or numpy array representing the predicted value.
	Return:
	cost -- cost for given parameters
	"""
	return (1/Y.shape[1]) * np.sum(((Y - A)))

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
	m = X.shape[1]
	dw = (1/m) * np.dot(X,((A - Y).T))
	db = (1/m) *  np.sum(((A - Y))) 
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
			print("\ncost after {0}th iteration: {1}".format(i, cost))
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
	#w = w.reshape(X.shape[1], 1)
	A = np.dot(w.T, X) + b
	return A	

def linear_regression(X_train, Y_train, X_test, Y_test, epochs = 2000, learning_rate = 0.005):
	"""
	Builds the logistic regression model by calling the function you've implemented previously

	Arguments:
	X_train -- training set represented by a numpy array of shape (number of features, m_train)
	Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	X_test -- test set represented by a numpy array of shape (number of features, m_test)
	Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	epochs -- hyperparameter representing the number of iterations to optimize the parameters
	learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
	
	Returns:
	d -- dictionary containing information about the model.
	"""
	w = np.zeros((X_train.shape[0],1))
	b = 0
	w, b, dw, db, costs = optimize(X_train, Y_train, w, b, epochs, learning_rate)
	Y_prediction_test = predict(X_test, w, b)
	Y_prediction_train = predict(X_train, w, b)

	# Print train/test Errors
	train_accuracy = (np.sqrt(np.sum((Y_prediction_train - Y_train)**2)/X_train.shape[1]))
	test_accuracy = (np.sqrt(np.sum((Y_prediction_test - Y_test)**2))/X_test.shape[0])
	print("\ntrain accuracy (mse): {} ".format(train_accuracy))
	print("test accuracy (mse): {} ".format(test_accuracy))

	d = {"costs": costs,
		"Y_prediction_test": Y_prediction_test, 
		"Y_prediction_train" : Y_prediction_train, 
		"w" : w, 
		"b" : b,
		"learning_rate" : learning_rate,
		"epochs": epochs,
		"train_accuracy": train_accuracy,
		"test_accuracy": test_accuracy}
	return d


def main():
	X_train, Y_train, X_test, Y_test = load_dataset()

	print(X_train.shape, Y_train.shape, X_test.shape, Y_train.shape)
	#NN(X,Y,test_set_x, test_set_y)
#	w, b, X, Y = np.array([[1.]]), 2., np.array([[1.,2.]]), np.array([[10,50]])
	linear_regression(X_train, Y_train, X_test, X_test)

if __name__ == '__main__':
	main()