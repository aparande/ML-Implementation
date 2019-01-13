"""
A Python implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
Works in both Python2 and Python3
"""
import numpy as np
import math
from mnist import load_data
import pickle

class ViolaJones:
    def __init__(self, feature_num = 10):
        """
          Args:
            feature_num: The number of weak classifiers which should be used
        """
        self.feature_num = feature_num
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
        print("Computing integral images")
        for x in range(len(training_data)):
            training_data[x] = (integral_image(training_data[x][0]), training_data[x][1])
            if training_data[x][1] == 1:
                weights[0][x] = 1.0 / (2 * pos_num)
            else:
                weights[0][x] = 1.0 / (2 * neg_num)

        print("Starting training")
        features = self.build_features(training_data[0][0].shape)

        for t in range(self.feature_num):
            print("Choosing classifier #%d" % (t+1))
            weights[t] = weights[t] / np.linalg.norm(weights[t])
            weak_classifiers = self.train_weak(features, weights[t], training_data)
            clf, error, accuracy = self.select_best(weak_classifiers, weights[t], training_data)
            beta = error / (1.0 - error)
            new_weights = np.zeros(weights[t].shape)
            for i in range(len(accuracy)):
                new_weights[i] = weights[t][i] * (beta ** accuracy[i])
            weights = np.vstack((weights, new_weights))
            self.alphas.append(math.log(1.0/beta))
            self.clfs.append(clf)

    def train_weak(self, features, weights, training_data):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: an array of weights for each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, ex in zip(weights, training_data):
            if ex[1] == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        for positive_regions, negative_regions in features:
            if len(classifiers) % 1000 == 0:
                print("Trained %d classifiers" % len(classifiers))
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
            training = map(lambda data: (feature(data[0]), data[1]), training_data)
            training = sorted(zip(weights, training), key=lambda ex: ex[1][0])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, ex in training:
                if ex[1] == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
                
                error = min(neg_weights+total_pos-pos_weights, pos_weights+total_neg-neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = (positive_regions, negative_regions)
                    best_threshold = ex[0]
                    best_polarity = 1 if pos_seen < neg_seen else -1
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
                
    def build_features(self, image_shape):
        """
        Builds the possible features given an image shape
          Args:
            image_shape: a tuple of form (height, width)
          Returns:
            an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
        """
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return features

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        best_clf, best_error, best_accuracy = classifiers[0], float('inf'), []
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'r') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity
          Args:
            x: A 2D numpy array of shape (m, n) representing the integral image
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        feature = feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

class RectangleRegion:
    def __init__(self, x, y, width, height):
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
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

if __name__ == "__main__":
    training, validation, test = load_data()
    training = zip(training[0], training[1])
    training = [(np.reshape(image, (28, 28)), label) for image, label in training if label == 1 or label == 0]
        
    try:
        clf = ViolaJones.load("viola_jones")
        print("Loaded classifier from file")
    except Exception as e:
        print("Failed to load classifier from file: %s" % str(e))
        pos_num, neg_num = 0, 0
        for x, y in training:
            if y == 1:
                pos_num += 1
            else:
                neg_num += 1
        clf = ViolaJones(feature_num=2)
        clf.train(training, pos_num, neg_num)

        #clf.save("viola_jones")

    correct = 0
    for x, y in training:
        prediction = clf.classify(x)
        correct += prediction == y
    print("Correctly labeled %d out of %d training examples" % (correct, len(training)))