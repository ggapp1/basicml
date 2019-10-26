import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

def load_dataset():
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(x, w, b):
	return 1/(1+np.exp(-1*(np.dot(w, x) + b)))

def gradientDescent(X, Y, w, b):
	Z = np.dot(w.transpose(), X) + b
	A = sigmoid(Z)
	#number of features
	m = len(X)
	dz = A - Y
	db = 1/m * sum(dz)
	dw = 1/m * np.dot(X, dz.T)  
	
	return dw, db

def logisticRegression(X, Y,n):
	w = np.random()
	b = np.random()
	#learnig rate
	alpha = 0.05

	for i in range(n):         	                        
		dw, db = gradientDescent(X, Y, w, b)
		w = w - alpha * dw
		b = b - alpha * db
	
	return w	
	
def main():
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
	# Example of a picture
	m_train = len(train_set_x_orig)
	m_test = len(test_set_x_orig)
	num_px = train_set_x_orig[0].shape[0]
	print ("Number of training examples: m_train = " + str(m_train))
	print ("Number of testing examples: m_test = " + str(m_test))
	print ("Height/Width of each image: num_px = " + str(num_px))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_set_x shape: " + str(train_set_x_orig.shape))
	print ("train_set_y shape: " + str(train_set_y.shape))
	print ("test_set_x shape: " + str(test_set_x_orig.shape))
	print ("test_set_y shape: " + str(test_set_y.shape))

if __name__ == '__main__':
	main()