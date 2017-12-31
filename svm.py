import numpy as np
from matplotlib import pyplot as plt

def showPlot(data, labels, w=None, b=0):
    for x in reversed(range(0, len(data))):
        if x > len(labels)-1 and w != None:
            if np.dot(w, data[x]) + b < 0:
                plt.scatter(data[x][0], data[x][1], s=120, marker='_', linewidths=2, color="blue")
            else:
                plt.scatter(data[x][0], data[x][1], s=120, marker='+', linewidths=2, color="blue")
            continue
        elif x > len(labels)-1:
            continue

        if labels[x] == -1:
            plt.scatter(data[x][0], data[x][1], s=120, marker='_', linewidths=2, color="red")
        else:
            plt.scatter(data[x][0], data[x][1], s=120, marker='+', linewidths=2, color="green")
    plt.show()

def step_descend(data, labels, startW=None, startB=0, training_rate=0.001, iter=10000):
    if startW is None:
        startW = np.zeros(len(data[0]))

    newW = startW
    newB = startB
    for i in range(0, iter):
        for j in range(0, len(data)):
            x = data[j]
            y = labels[j]
            if y*(np.dot(x, newW) + newB) < 1:
                #Incorrect classification
                newW = newW - training_rate*(newW - y*x)
                newB = newB - training_rate
            else:
                #Correct classification
                newW = newW - training_rate * newW
                newB = newB + training_rate

    return newW, newB

X = np.array([
    [-2,4],
    [4,1],
    [1, 6],
    [2, 4],
    [6, 2]
])

labels = np.array([-1,-1,1,1,1])

w, b = step_descend(X, labels)
print (w)
print (b)

for i in range(-2, 10):
    for j in range(-2, 10):
        X = np.append(X, [[i, j]], axis=0)

showPlot(X, labels, w=w, b=b)
