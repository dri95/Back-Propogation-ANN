""" 
The designed feedforward neural network uses back propogation algorithm to 
updates its weight and is further optimized using momentum and also uses 
Regularization for the optimization the cost function.

Python Version 3.6.5
Libraries Used : Pandas, Numpy and sklearn.metrices 
"""

import numpy as np
from sklearn.metrics import accuracy_score


class Backprop_ANN(object):
	"""
	The initialization function takes the following parameters:
	1)layers: layers for the FF network in a list (eg. [8 4 1]).
	2)learning_rate: learning rate of the network (c).
	3)momentum: momentum parameter (alpha).
	"""
	def __init__(self, layers, learning_rate, momentum, regularization):
		
		self.c = learning_rate
		self.alpha = momentum
		self.input_count = layers[0]
		self.output_count = layers[-1]
		self.No_layers = len(layers)
		self.lam = regularization
		self.weights = []
		self.biases = []
		for i in range(1, self.No_layers):
			previous_outputs = layers[i-1]
			initial_parameter = 1.0 / previous_outputs
			"""
			initialize weight matrix using uniform initialization with dimensions (n x m)
			and a (n x 1) bias vector.
			"""
			weight_mtx = np.random.uniform(-initial_parameter, initial_parameter, size=(layers[i], previous_outputs)) 
			bias_vector = np.zeros((layers[i], 1))	
			self.weights.append(weight_mtx)
			self.biases.append(bias_vector)
    
	"""
	The one_hot functions performs executes encoding of numeric lable.
	"""
	def one_hot(self, d):
		y = np.zeros((self.output_count, 1))
		y[d] = 1
		return y
    
	"""
	The train function takes the following parameters:
	1)train_X: training dataset set
	2)train_D: labels of training dataset
	3)val_X:  validation dataset
	4)val_D: labels of validation dataset
	5)No_epochs: No of iterations 
	"""
	def network_train(self, train_X, train_Y, val_X=None, val_Y=None, No_epochs=1):
		train_Y = list(map(self.one_hot, train_Y))
		accuracies = []
		for epoch in range(No_epochs):
			for x, y in zip(train_X, train_Y):
				"""
				updating weights
				"""
				self.update_weights(x.reshape(1, self.input_count), y)
			if val_X is not None and val_Y is not None:
				predictions = self.network_test(val_X)
				accuracies.append(accuracy_score(val_Y, predictions))
		"""
		Return validation accuracy
		"""        
		return accuracies
    
	"""
	The function network_test returns predicted labels for test dataset
	"""
	def network_test(self, test_X):
		return list(map(self.predict, test_X))
    
	"""
	The update_weights function takes in x: the input feature pattern vector and 
	y: the correcponding lable for the vector and updates the weights using gradient
	decent after each input pattern.
	"""
	def update_weights(self, x, y):
		del_w, del_b = self.backpropogate(x, y)
		self.weights = [w - self.c * dw + self.alpha * w for w, dw in zip(self.weights, del_w)]
		self.biases = [b - self.c * db + self.alpha * b for b, db in zip(self.biases, del_b)]
		return None
    
	"""
	The backpropogate function implements the bacprogation algorithm and takes in
	the same parameters as the update_weights function.
	"""    
	def backpropogate(self, x, y):
		"""
		del_w and del_b are the gradients of the cost function w.r.t. the weights and biases
		"""
		del_w = [np.zeros(w.shape) for w in self.weights]
		del_b = [np.zeros(b.shape) for b in self.biases]
		"""
		Foward pass through the network as per the lecture slides
		"""
		activation = x.T
		activations = [activation]
		net_inputs = []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			net_inputs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		"""
		Calculating the error in the output layer
		"""    
		output = self.one_hot(np.argmax(activations[-1]))
		delta = self.cost_derivative(output, y) * sigmoid_prime(net_inputs[-1]) #dC/dz at output
		del_b[-1] = delta #dC/db = dC/dz
		del_w[-1] = np.dot(delta, activations[-2].T)
		"""
		propogate error backwards layer by layer in the network
		"""
		for l in range(2, self.No_layers):
			z = net_inputs[-l]
			delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_prime(z)
			del_b[-l] = delta
			del_w[-l] = np.dot(delta, activations[-l-1].T)
		return (del_w, del_b)
    
	"""
	The network_predict function maps the inputs to the outputs depending on the activation
	rule. It predicts a lable given a feature imput vector.
	"""
	def predict(self, x):
		activation = x.reshape(1, self.input_count).T
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b 
			activation = sigmoid(z)
		return np.argmax(activation)    
     
	"""
	Cost Function takes the reqularization paramater(lam) 
	"""
	@staticmethod
	def cost_function(self,output_activations, y):
		return self.lam * (y - output_activations) ** 2
    
	"""
	derivative of the cost function
	"""
	@staticmethod
	def cost_derivative(output_activations, y):
		return (output_activations - y)

"""
The entire network uses sigmoid activation function
"""        
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

"""
Derivative of the sigmoid function
""" 

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))
"""
Stats
""" 
def precision_recall(confusion_matrix):
	result = []
	for i in range(confusion_matrix.shape[0]):
		precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
		recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
		result.append({'precision' : precision, 'recall' : recall})
	return result   
