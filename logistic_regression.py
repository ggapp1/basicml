import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

def load_dataset():
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
	return 1/(1+np.exp(np.dot(-1,Z)))

def cost_function(Y, A):
	return (-1/Y.shape[1]) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1- Y, np.log(1 - A)))

def gradient_descent(X, Y, w, b):
	A = sigmoid(np.dot(w.T, X) + b)	
	#number of features
	m = X.shape[0]
	dz = A - Y
	db = 1/m * np.sum(dz)
	dw = 1/m * np.dot(X, dz.T) 

	return dw, db, cost_function(Y, A)

def optimize(X, Y, w, b, epochs, learning_rate):
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
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)	
	A[A <= 0.5] = 0
	A[A > 0.5] = 1

	return A	

def logistic_regression(X_train, Y_train, X_test, Y_test, epochs = 2000, learning_rate = 0.5):
	w = np.zeros((X_train.shape[0],1))
	b = 0

	w, b, dw, db, costs = optimize(X_train, Y_train, w, b, epochs, learning_rate)

	Y_prediction_test = predict(X_test, w, b)
	Y_prediction_train = predict(X_train, w, b)

	# Print train/test Errors
	print("\ntrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
	d = {"costs": costs,
		"Y_prediction_test": Y_prediction_test, 
		"Y_prediction_train" : Y_prediction_train, 
		"w" : w, 
		"b" : b,
		"learning_rate" : learning_rate,
		"epochs": epochs}
	return d

def plot_learning_curve(model):
	# Plot learning curve (with costs)
	costs = np.squeeze(model['costs'])
	plt.plot(costs)
	plt.title('Learning curve')
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(model["learning_rate"]))
	plt.show()

def test_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y):
	learning_rates = [0.01, 0.001, 0.0001, 0.05, 0.005]
	models = {}
	for i in learning_rates:
	    print ("learning rate is: " + str(i))
	    models[str(i)] = logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y, epochs = 1500, learning_rate = i)
	    print ('\n' + "-------------------------------------------------------" + '\n')

	for i in learning_rates:
	    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

	plt.title('Learning curve per Learning rate')	    
	plt.ylabel('cost')
	plt.xlabel('iterations (hundreds)')
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()

def main():
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	m_train = len(train_set_x_orig)
	m_test = len(test_set_x_orig)
	print ("Number of training examples: m_train = " + str(m_train))
	print ("Number of testing examples: m_test = " + str(m_test))
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

	train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255.
	test_set_x =  test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255.

	model = logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y)
	plot_learning_curve(model)
	test_learning_rates(train_set_x, train_set_y, test_set_x, test_set_y)


if __name__ == '__main__':
	main()