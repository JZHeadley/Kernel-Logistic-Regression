import numpy as np


def flat_norm(dataset):
    return np.array(normalize(flatten(dataset)))


def normalize(dataset):
    for i in range(0, dataset.__len__()):
        for j in range(0, dataset[i].__len__()):
            if dataset[i][j] > 0:
                dataset[i][j] = float(dataset[i][j] / float(255))
    return dataset


def flatten(arrayOfMatrix):
    flattened = []
    for i in range(0, arrayOfMatrix.__len__()):
        flat_x = []
        for j in range(0, arrayOfMatrix[i].__len__()):
            for k in range(0, arrayOfMatrix[i][j].__len__()):
                flat_x.append(arrayOfMatrix[i][j][k])
        flattened.append(flat_x)
    return flattened


def extractClass(x, y, toExtract):
    classSamples = []
    for i in range(0, x.__len__()):
        if y[i] == toExtract:
            classSamples.append(x[i])
    return classSamples


def computeAccuracy(trueY, predY):
    numElements = trueY.__len__()
    correctPredictions = 0
    for i in range(0, numElements):
        correctPredictions += (trueY[i] == predY[i])
    return (correctPredictions/numElements)


def extractMine(x, y, class1, class2):
    x_mine = []
    y_mine = []
    for i in range(0, x.__len__()):
        if y[i] == class1 or y[i] == class2:
            x_mine.append(x[i])
            if(y[i] == class1):
                y_mine.append(1)
            elif(y[i] == class2):
                y_mine.append(-1)
    return (np.array(x_mine), np.array(y_mine).reshape(y_mine.__len__(), 1))
