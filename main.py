import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plotImg(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def drawPoints(img, points, labels):
    for index, point in enumerate(points):
        i, j = point
        isCorner = True if labels[index] == 1 else False

        if isCorner:
            cv2.circle(img,(j, i), 5, (0, 0, 255))
        else:
            cv2.circle(img,(j, i), 5, (255, 255, 255))

def selectPoints(harrisImg, maxNumOfKeyPoints, thresholdRate = 0.01):
    cornerPoints = []
    nonCornerPoints = []

    row, col = harrisImg.shape
    maxVal = harrisImg.max()

    # detect corners
    for i in range(row):
        for j in range(col):
            if harrisImg[i][j] > maxVal * thresholdRate:
                harrisImg[i][j] = 0
                cornerPoints.append([i, j])

    # detect non corner points
    for x in range(maxNumOfKeyPoints):
        i = random.randint(0, row - 1)
        j = random.randint(0, col - 1)

        if harrisImg[i][j] != 0:
            nonCornerPoints.append([i, j])

    return cornerPoints, nonCornerPoints

def selectSamplePoints(points, lenSample, testRate = 0.1):
    # select arbitrary elements up to sample length
    sampledPoints = random.sample(points, lenSample)
    sampleTrain, sampleTest = train_test_split(sampledPoints, test_size = testRate)

    return sampleTrain, sampleTest

def determineSampleLength(cornerPoints, nonCornerPoints):
    lenCorner = len(cornerPoints)
    lenNonCorner = len(nonCornerPoints)

    lenSample = lenCorner
    if lenNonCorner < lenCorner:
        lenSample = lenNonCorner
    
    return lenSample

def createLabels(cornerPoints, nonCornerPoints):
    lenCorner = len(cornerPoints)
    lenNonCorner = len(nonCornerPoints)

    cornerTrainLabels = [1 for i in range(lenCorner)]
    nonCornerTrainLabels = [-1 for i in range(lenNonCorner)]

    labels = cornerTrainLabels + nonCornerTrainLabels
    return labels

def shuffleLists(pointsX, pointsY):
    pointsZipList = list(zip(pointsX, pointsY))
    random.shuffle(pointsZipList)

    shuffledPointsX, shuffledPointsY = zip(*pointsZipList)
    return shuffledPointsX, shuffledPointsY

def getKeyPoints(imgGray, shuffle = True):
    maxNumOfKeyPoints = 2000

    # run harris corner detector on the img
    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)

    # get corner and non corner points on the image up to the given number
    cornerPoints, nonCornerPoints = selectPoints(dst, maxNumOfKeyPoints, thresholdRate = 0.1)

    # determine sample length to select this number of elements
    lenSample = determineSampleLength(cornerPoints, nonCornerPoints)

    # get train and test samples
    cornerTrain, cornerTest = selectSamplePoints(cornerPoints, lenSample, testRate = 0.1)
    nonCornerTrain, nonCornerTest = selectSamplePoints(nonCornerPoints, lenSample, testRate = 0.1)

    # concatenate points
    trainPoints = cornerTrain + nonCornerTrain
    testPoints  = cornerTest + nonCornerTest

    # create and get concatenated labels
    trainLabels = createLabels(cornerTrain, nonCornerTrain)
    testLabels  = createLabels(cornerTest , nonCornerTest)

    # shuffle the points if required
    if shuffle == True:
        trainPoints, trainLabels = shuffleLists(trainPoints, trainLabels)
        testPoints,  testLabels = shuffleLists(testPoints, testLabels)

    return trainPoints, trainLabels, testPoints, testLabels

def gradientsOfRegions(img, points, regionSize = 7):
    borderedImg = cv2.copyMakeBorder(img, regionSize // 2, regionSize // 2, regionSize // 2, regionSize // 2, cv2.BORDER_CONSTANT)

    featureVectors = []
    for point in points:
        i, j = point

        # shift the points to the true points, before the padding
        i, j = i + regionSize // 2, j + regionSize // 2

        region = borderedImg[i - regionSize // 2 : i + regionSize // 2 + 1, j - regionSize // 2 : j + regionSize // 2 + 1]

        # calculate gradients
        gradX = cv2.Sobel(region, cv2.CV_32F, 1, 0)
        gradY = cv2.Sobel(region, cv2.CV_32F, 0, 1)

        # reshape gradient list to be a 1D list
        gradX = np.reshape(gradX, regionSize * regionSize)
        gradY = np.reshape(gradY, regionSize * regionSize)

        # calculate the magnitude of the gradients
        magnitude, _ = cv2.cartToPolar(gradX, gradY)
        magnitude = np.reshape(magnitude, regionSize * regionSize)

        # concatenate 1D gradients and their magnitude to be a feature vector
        featureVector = list(gradX) + list(gradY) + list(magnitude)

        featureVectors.append(featureVector)

    return featureVectors

import os
def createTrainTestData(imgDir):
    directory = os.listdir(imgDir)

    allTrainVectors, allTestVectors = [], []
    allTrainLabels, allTestLabels = (), ()
    for fileName in directory:
        path = os.path.join(imgDir, fileName)

        img = cv2.imread(path)
        blurredImg = cv2.GaussianBlur(img, (5, 5), 0)
        imgGray = cv2.cvtColor(blurredImg, cv2.COLOR_RGB2GRAY)

        trainPoints, trainLabels, testPoints, testLabels = getKeyPoints(imgGray, shuffle = True)
        trainVectors = gradientsOfRegions(imgGray, trainPoints)
        testVectors = gradientsOfRegions(imgGray, testPoints)

        allTrainVectors += trainVectors
        allTrainLabels += trainLabels

        allTestVectors += testVectors
        allTestLabels  += testLabels

    return np.array(allTrainVectors, np.float32), np.array(allTrainLabels), np.array(allTestVectors, np.float32), np.array(allTestLabels)

imgDir = "./img"
trainX, trainY, testX, testY = createTrainTestData(imgDir)

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm.train(trainX, cv2.ml.ROW_SAMPLE, trainY)
predictions = svm.predict(testX)[1]

cm = confusion_matrix(testY, predictions, labels = [1, -1])
tn, fp, fn, tp = cm.ravel()
print(cm)
print((tn, fp, fn, tp))
print((tp + tn) / len(testY))