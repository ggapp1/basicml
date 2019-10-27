import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import pandas as pd

def load_dataset():
	"""
	Load the cat/non-cat dataset

	Return:
	train_set_x_orig, train_set_y_orig -- (x,y) of training set
	test_set_x_orig, test_set_y_orig, -- (x,y) of testing set
	classes -- classes on dataset
	"""	
	with h5py.File('../datasets/train_catvnoncat.h5', "r") as train_dataset:
		train_set_x_orig = np.array(train_dataset["train_set_x"][:])
		train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	with h5py.File('../datasets/test_catvnoncat.h5', "r") as test_dataset:
		test_set_x_orig = np.array(test_dataset["test_set_x"][:])
		test_set_y_orig = np.array(test_dataset["test_set_y"][:])
		classes = np.array(test_dataset["list_classes"][:])

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(Z):
	"""
	Compute the sigmoid of z (1/(1+e^-z))

	Arguments:
	z -- A scalar or numpy array of any size.

	Return:
	s -- sigmoid(z)
	"""
	return 1/(1+np.exp(np.dot(-1,Z)))

def cost_function(Y, A):
	"""
	Compute the loss function (negative log-likelihood cost for logistic regression)

	Arguments:
	Y -- A scalar or numpy array represeting the actual label.
	A -- A scalar or numpy array representing the predicted label.
	Return:
	cost -- cost for given parameters
	"""
	return (-1/Y.shape[1]) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1- Y, np.log(1 - A)))

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
	A = sigmoid(np.dot(w.T, X) + b)	
	#number of features
	m = X.shape[0]
	dz = A - Y
	db = 1/m * np.sum(dz)
	dw = 1/m * np.dot(X, dz.T) 

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
	Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	'''
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)	
	A[A <= 0.5] = 0
	A[A > 0.5] = 1

	return A	

def logistic_regression(X_train, Y_train, X_test, Y_test, epochs = 2000, learning_rate = 0.5):
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
	train_accuracy = (100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)
	test_accuracy = (100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)
	print("\ntrain accuracy: {} %".format(train_accuracy))
	print("test accuracy: {} %".format(test_accuracy))

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

def plot_learning_curve(model):
	"""
	Plot the learning curve for a specific model

	Arguments:
	model -- a logistic regression model
	"""
	costs = np.squeeze(model['costs'])
	plt.plot(costs)
	plt.title('Learning curve')
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(model["learning_rate"]))
	plt.show()

def test_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y):
	"""
	Plot the learning curve and the accuracy varying the learning rate

	Arguments:
	X_train -- training set represented by a numpy array of shape (number of features, m_train)
	Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	X_test -- test set represented by a numpy array of shape (number of features, m_test)
	Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	"""
	learning_rates = [0.01, 0.001, 0.0001, 0.05, 0.005]
	train_accuracy = []
	test_accuracy = []
	models = {}
	for i in learning_rates:
		print ("learning rate is: " + str(i))
		models[str(i)] = logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y, epochs = 1500, learning_rate = i)
		train_accuracy.append(models[str(i)]["train_accuracy"])
		test_accuracy.append(models[str(i)]["test_accuracy"])
		print ('\n' + "-------------------------------------------------------" + '\n')

	for i in learning_rates:
		plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

	#plot learning curves
	plt.title('Learning curve per Learning rate')		
	plt.ylabel('cost')
	plt.xlabel('iterations (hundreds)')
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()

	#plot accuracy bar graph
	df = pd.DataFrame({"Train accuracy":train_accuracy,"Test accuracy":test_accuracy})
	ax = df.plot.bar(color=["SkyBlue","LightGreen"], rot=0, title="Accuracy per learning rate")
	ax.set_xlabel("Learning rate")
	ax.set_ylabel("Accuracy")
	ax.xaxis.set_major_formatter(plt.FixedFormatter(learning_rates))
	plt.show()

def subcategorybar(X, vals, width=0.8):
	n = len(vals)
	label = ['train', 'test']
	_X = np.arange(len(X))
	for i in range(n):
		plt.bar(_X - width/2. + i/float(n)*width, vals[i], label='inferno',
				width=width/float(n), align="edge")   
	plt.xticks(_X, X)

def test_epochs(train_set_x, train_set_y, test_set_x, test_set_y):
	"""
	Plot the  accuracy varying the number of epochs
	Arguments:
	X_train -- training set represented by a numpy array of shape (number of features, m_train)
	Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	X_test -- test set represented by a numpy array of shape (number of features, m_test)
	Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	"""
	epochs = [100, 500, 1000, 2000, 5000, 10000]
	models = {}
	train_accuracy = []
	test_accuracy = []

	for i in epochs:
		print ("No of epochs: " + str(i))
		models[str(i)] = logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y, epochs = i)
		train_accuracy.append(models[str(i)]["train_accuracy"])
		test_accuracy.append(models[str(i)]["test_accuracy"])
		print ('\n' + "-------------------------------------------------------" + '\n')

	#plot accuracy bar graph
	df = pd.DataFrame({"Train accuracy":train_accuracy,"Test accuracy":test_accuracy})
	ax = df.plot.bar(color=["SkyBlue","LightGreen"], rot=0, title="Accuracy per number of epochs")
	ax.set_xlabel("Number of epochs")
	ax.set_ylabel("Accuracy")
	ax.xaxis.set_major_formatter(plt.FixedFormatter(epochs))
	plt.show()

def main():
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	m_train = len(train_set_x_orig)
	m_test = len(test_set_x_orig)
	print ("Number of training examples: m_train = " + str(m_train))
	print ("Number of testing examples: m_test = " + str(m_test))
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

	#reshape dataset
	train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255.
	test_set_x =  test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255.

	model = logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y)
	plot_learning_curve(model)
	test_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y)
	test_epochs(train_set_x, train_set_y, test_set_x, test_set_y)

if __name__ == '__main__':
	main()