"""
A custom class for a Vanilla Neural Network
Made with guidance from http://neuralnetworksanddeeplearning.com/chap1.html
"""

import numpy as np
import random

class NeuralNetwork:

    def __init__(self, layer_dims, regularizer = 0, cost_function = "CrossEntropy"):
        self.cost = cost_function
        self.layer_sizes = layer_dims
        self.regularizer = regularizer
        #Only the layers 1 and above will have biases
        self.biases = [np.random.randn(x, 1) for x in layer_dims[1:]]
        #The tuples of the array containing all but the output layer and the array containing all but the input layer gives the size of the weight matrices
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for (x, y) in zip(layer_dims[:-1], layer_dims[1:])]

    #Gives the derivative of the cost function
    def get_delta(self, a, y, z):
        if self.cost == "CrossEntropy":
            return a - y
        else:
            #The quadratic cost
            return np.multiply((a - y), sigmoid(z, True))

    #will get the gradient for a single training example
    def backprop(self, x, y):
        #Keep track of zs and activations to make calculations easier
        zs = []
        #Activations of the input layer
        a = x #Transpose the x because of the formatting of the data
        activations = [a] #The activtions of the input layer is always a. Makes the following loop cleaner
        #Feed-forward step to get all the z's and activation values so we can later calculate the error_deltas
        for (w, b) in zip(self.weights, self.biases):
            #Calculate the z vector using the weight and bias vectors
            z = np.dot(w, a) + b
            #Calculate the activation using the z value
            a = sigmoid(z)
            #Book-keeping step
            zs.append(z)
            activations.append(a)

        #Get the error delta for the last layer
        error_delta = self.get_delta(activations[-1], y, zs[-1])

        #Store each error_delta to store the gradient for the training example.
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        #Calculate the gradient for the last layer
        grad_w[-1] = np.dot(error_delta, activations[-2].transpose()) #Use -2 instead of -1 because the formula is the l-1 layer
        grad_b[-1] = error_delta

        #Backpropagate using negative indexing
        for i in range(2, len(self.layer_sizes)):
            w = self.weights[-i+1] #It's -i + 1 since the weight layer we use is the l + 1 weight layer
            z_prime = sigmoid(zs[-i], True)

            error_delta = np.multiply(np.dot(w.transpose(), error_delta), z_prime)
            grad_w[-i] = np.dot(error_delta, activations[-i-1].transpose())
            grad_b[-i] = error_delta
        #Weight gradient, bias gradient
        return (grad_w, grad_b)

    #A single traning step consists of updating the weights and biases once
    #n is the total number of points in the dataset
    def training_step(self, mini_batch, training_rate, n):
        m = len(mini_batch)

        #Store the gradients
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        #Calculate the gradient vector
        for (x, y) in mini_batch:
            grad_w_x, grad_b_x = self.backprop(x, y)
            #Add all of the gradients for each training example to the overall gradient
            grad_w = [gw + gx for (gw, gx) in zip(grad_w, grad_w_x)]
            grad_b = [gb + gx for (gb, gx) in zip(grad_b, grad_b_x)]

        #The approximation of the gradient is the average of the gradients for the training data, so divide each grad_x by the training count
        #Slow training with the training_rate
        #Update the weights and biases by descending the gradient
        regularization_coeff = 1 - training_rate*self.regularizer/n
        self.weights = [regularization_coeff*w - training_rate/m * gw for (w, gw) in zip(self.weights, grad_w)]
        self.biases = [b - training_rate/m * gb for (b, gb) in zip(self.biases, grad_b)]

    def classify(self, x):
        activation = x
        for (w, b) in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)

        return activation

    #Training data will be a tuple containing the feature and the label
    #Test data can be used to evaluate the networks capabilities as it trains
    def train(self, training_data, batch_size=50, epochs = 100, training_rate = 0.01, test_data = None, verbose=False):
        for x in range(1, epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i+batch_size] for i in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.training_step(batch, training_rate, len(training_data))

            if verbose:
                if test_data:
                    print ("Epoch {0} complete: {1} correct out of {2}".format(x, self.evaluate(test_data), len(test_data)))
                else:
                    print ("Epoch {0} complete".format(x))

    #Return the number of test inputs that the network correctly classifies
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.classify(x)), y) for (x, y) in test_data]
        return sum (int(x == y) for (x, y) in test_results)

def sigmoid(z,deriv=False):
    if(deriv==True):
        return np.multiply(sigmoid(z), (1-sigmoid(z)))

    return 1/(1+np.exp(-z))
