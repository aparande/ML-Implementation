"""
A custom implementation of a binary logistic classifier written in python 2.7.10
"""

import numpy as np
class LogisticRegression:

    def __init__(self, input_number, lr=0.0001):
        self.data_dimension = input_number
        self.learning_rate = lr

        #There will be one weight for each dimension of data
        self.weights = np.zeros((1, self.data_dimension))
        self.bias = 0

    """
    Calculates the cross entropy cost

    Arguments:
        training_data: A numpy array with dimension (number of examples, data_dimension)
        training_labels: A numpy array with dimension (number of examples, 1)

    Returns:
        The cross entropy cost
    """
    def loss(self, training_data, training_labels):
        z = np.dot(training_data, self.weights.T) + self.bias
        prob = self.sigmoid(z)

        cross_entropy = np.multiply(training_labels, np.log(prob)) + np.multiply(1-training_labels, np.log(1-prob))
        return -np.mean(cross_entropy)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    """
    Train the model with the provided data and labels

    Arguments:
        training_data: A numpy array with dimension (number of examples, data_dimension)
        training_labels: A numpy array with dimension (number of examples, 1)
        epochs: How many epochs the algorithm should run for
        verbose: Should the algorithm print out loss periodically
    """
    def train(self, training_data, training_labels, epochs = 10000, verbose=True):
        data_len = len(training_data)

        for step in range(epochs):
            grad_w = np.zeros((1, self.data_dimension))
            grad_b = 0
            for (x, y) in zip(training_data, training_labels):
                z = np.dot(self.weights, x) + self.bias

                #Calculate the gradient for the training example. 
                grad_w += (self.sigmoid(z) - y) * x
                grad_b += (self.sigmoid(z) - y)

            #Subtract because you are going in the opposite direction of the gradient
            self.weights -= self.learning_rate/data_len * grad_w
            self.bias -= self.learning_rate/data_len * grad_b

            if (step % 500 == 0):
                print "Loss at step " + str(step) + ": " + str(self.loss(training_data, training_labels))
    """
    Classify the set of training examples
    Arguments:
        test_data: A numpy array with dimension (data length, data_dimension)
        threshold: Where the classifier should draw the line between class 0 and class 1
    Returns:
        classification: A numpy array with dimension (data length, 1)
    """
    def classify(self, test_data, test_labels, threshold=0.5, ):
        classification = np.zeros((1, len(test_data)))
        
        index = 0
        for x in test_data:
            z = np.dot(self.weights, x) + self.bias
            prob = self.sigmoid(z)
            if (prob > threshold):
                classification[0][index] = 1
            index += 1

        correct = 0
        for x in range(len(test_labels)):
            if (classification[0][x] == test_labels[x]):
                correct = correct + 1
        
        print "Accuracy: " + str(correct) + " out of "+str(len(test_data))

        return classification