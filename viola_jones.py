"""
A Python3 implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
"""
import numpy as np
from neuralnetwork import NeuralNetwork as NN
import math

class ViolaJones:
    def __init__(self, feature_num = 10, learning_rate=0.01, regularizer = 0):
        self.feature_num = feature_num
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.alphas = []
        self.clfs = []

    def train(self, training_data, pos_num, neg_num):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
          Args:
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        weights = np.zeros((1, len(training_data)))
        for x in len(training_data):
            training_data[x][0] = integral_image(training_data[x][0])
            if training_data[x][1] == 1:
                weights = 1 / (2 * pos_num)
            else:
                weights = 1 / (2 * neg_num)

        
        for t in range(self.feature_num):
            weights[t] = np.linalg.norm(weights[t], axis=1)
            clf, error, accuracy = self.select_weak(training_data, weights[t])
            beta = error / (1 - error)
            new_weights = np.zeros(weights[t].shape)
            for i in range(len(accuracy)):
                new_weights[i] = weights[t][i] * (beta ** accuracy[i])
            weights.vstack(new_weights)
            self.alphas.append(math.log(1/beta))
            self.clfs.append(clf)

    def classify(self, image):
        """
        Classifies an image
        """
        assert len(self.clfs) == self.feature_num, "Classifier is not trained"
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            feature = clf.feature(ii)
            total += alpha * np.argmax(clf.classify(feature))
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def train_weak(self, training_data, feature, weights):
        """
        Trains a weak classifier on the feature
          Args:
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
            feature: A single argument lambda which takes in an integral image and computes the feature
            weights: An array of weights for each training example
          Returns:
            The weak classifier and its error
        """
        training_data = map(lambda data: (feature(data[0]), one_hot(2, data[1])), training_data)
        clf = NN([1, 2], regularizer = self.regularizer)
        clf.feature = feature
        clf.train(training_data, epochs=5, training_rate=self.learning_rate)
        error = 0
        accuracy = []
        for data, w in zip(training_data, weights):
            correctness = abs(np.argmax(clf.classify(data[0])) - np.argmax(data[1]))
            accuracy.append(correctness)
            error += w * correctness
        return clf, error, accuracy

    def select_weak(self, training_data, weights):
        """
        Selects the weak classifier which produced the lowest error for a single feature
          Args:
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            weights: An array of weights for each training example
          Returns:
            A tuple where the first element is the classifier, the second is the error, and the third is the accuracy
        """
        height, width = training_data[0][0].shape
        classifiers = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w <= width:
                    j = 0
                    while j + h <= height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h).compute_feature
                        right = RectangleRegion(i+w, j, w, h).compute_feature
                        if i + 2 * w <= width: #Horizontally Adjacent
                            feature = lambda ii: right(ii) - immediate(ii)
                            classifiers.append(self.train_weak(training_data, feature, weights))

                        bottom = RectangleRegion(i, j+h, w, h).compute_feature
                        if j + 2 * h <= height: #Vertically Adjacent
                            feature = lambda ii: abs(immediate(ii) - bottom(ii))
                            classifiers.append(self.train_weak(training_data, feature, weights))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h).compute_feature
                        #3 rectangle features
                        if i + 3 * w <= width: #Horizontally Adjacent
                            feature = lambda ii: right(ii) - immediate(ii) - right_2(ii)
                            classifiers.append(self.train_weak(training_data, feature, weights))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h).compute_feature
                        if j + 3 * h <= height: #Vertically Adjacent
                            feature = lambda ii: bottom(ii) - immediate(ii) - bottom_2(ii)
                            classifiers.append(self.train_weak(training_data, feature, weights))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h).compute_feature
                        if i + 2 * w <= width and j + 2 * h <= height:
                            feature = lambda ii: right(ii) + bottom(ii) - immediate(ii) - bottom_right(ii)
                            classifiers.append(self.train_weak(training_data, feature, weights))

                        j += 1
                    i += 1
        return min(classifiers, key=lambda x: x[1])
        
class RectangleRegion:
    def __init__(self, x, y, width, height):
        assert x >= 0 and y >= 0 and width > 0 and height > 0, "Invalid Dimensions"
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        assert self.y < len(ii) and self.y + self.height < len(ii), "Invalid Region for image"
        assert self.x < len(ii[0]) and self.x + self.width < len(ii[0]), "Invalid Region for image"
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])
        
def integral_image(image):
    """
    Computes the integral image representation of a picture. The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image, and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Args:
        image : an numpy array with shape (m, n)
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in len(image):
        for x in len(image[y]):
            s[y][x] = s[y-1][x] + ii[y][x] if y-1 >= 0 else ii[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

def one_hot(length, label):
    """
    Transforms a numeric label into a one-hot vector
      Args:
        length: number of classes
        label: the classification
      Returns:
        One-hot vector representation of the label
    """
    e = np.zeros((length, 1))
    e[label] = 1.0
    return e


