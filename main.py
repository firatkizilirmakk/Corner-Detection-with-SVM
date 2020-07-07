import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import sys, getopt

def plotImg(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plotComparison(predicted, harris):
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(predicted)
    ax[1].imshow(harris)

    ax[0].set_xlabel("Predicted Corner Points")
    ax[1].set_xlabel("Harris Corner Points")

    plt.show()

def drawPoints(img, points, labels, drawOnlyCorners = False):
    for index, point in enumerate(points):
        i, j = point
        isCorner = True if labels[index] == 1 else False

        if isCorner:
            cv2.circle(img,(j, i), 5, (0, 0, 255))
        elif not drawOnlyCorners:
            cv2.circle(img,(j, i), 5, (0, 0, 0))

def selectCornerPoints(harrisImg, thresholdRate):
    row, col = harrisImg.shape
    maxVal = harrisImg.max()

    cornerPoints = []

    # detect corners
    for i in range(row):
        for j in range(col):
            if harrisImg[i][j] > maxVal * thresholdRate:
                harrisImg[i][j] = -1
                cornerPoints.append([i, j])

    return cornerPoints

def selectNonCornerPoints(harrisImg, imgGray, maxNumOfKeyPoints, regionSize):
    nonCornerPoints = []

    # detect non corner points by looking the near region around random points
    # select the point as a non corner if it is a flat region
    x = 0
    nonCornerThreshold = 1500
    while x < maxNumOfKeyPoints:

        # get random points
        i = random.randint(0, row - 1)
        j = random.randint(0, col - 1)

        # region around the point
        region = imgGray[i - regionSize // 2 : i + regionSize // 2 + 1, j - regionSize // 2 : j + regionSize // 2 + 1]
        regionRow, regionCol = region.shape

        # check region is valid
        if regionRow == regionSize and regionCol == regionSize and harrisImg[i][j] != -1:
            # calculate sobel gradients of the region
            gradX = cv2.Sobel(region, cv2.CV_32F, 1, 0)
            gradY = cv2.Sobel(region, cv2.CV_32F, 0, 1)

            #  sum up the magnitude of the gradients
            magnitude, _ = cv2.cartToPolar(gradX, gradY)
            sumMagnitude = np.sum(magnitude)

            # compare summed magnitude value with threshold to decide
            # whether the point is non corner
            if sumMagnitude < nonCornerThreshold:
                harrisImg[i][j] = -1
                nonCornerPoints.append([i, j])
                x += 1

    return nonCornerPoints

def selectPoints(harrisImg, imgGray, maxNumOfKeyPoints, regionSize ,thresholdRate = 0.01):

    # selects corner and non corner points
    # selects corner from the image applied harris
    # selects non corner from gray image by looking gradients
    cornerPoints = selectCornerPoints(harrisImg, thresholdRate)
    nonCornerPoints = selectNonCornerPoints(harrisImg, imgGray, maxNumOfKeyPoints, regionSize)

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

def getKeyPoints(imgGray, regionSize, shuffle = True):
    maxNumOfKeyPoints = 2000

    # run harris corner detector on the img
    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)

    # get corner and non corner points on the image up to the given number
    cornerPoints, nonCornerPoints = selectPoints(dst, imgGray, maxNumOfKeyPoints, regionSize, thresholdRate = 0.01)

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

def calculateFeatureVectors(img, points, regionSize):
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
def createTrainTestData(imgDir, regionSize):
    directory = os.listdir(imgDir)

    allTrainVectors, allTestVectors = [], []
    allTrainLabels, allTestLabels = (), ()
    for fileName in directory:
        path = os.path.join(imgDir, fileName)

        # read an image, apply gaussian and convert to grayscale
        img = cv2.imread(path)
        blurredImg = cv2.GaussianBlur(img, (5, 5), 0)
        imgGray = cv2.cvtColor(blurredImg, cv2.COLOR_RGB2GRAY)

        # get training and testing points and corresponding labels
        trainPoints, trainLabels, testPoints, testLabels = getKeyPoints(imgGray, regionSize = regionSize, shuffle = True)

        # calculate feature vectors on both training and testing points
        trainVectors = calculateFeatureVectors(imgGray, trainPoints, regionSize = regionSize)
        testVectors = calculateFeatureVectors(imgGray, testPoints, regionSize = regionSize)

        # append vectors and labels
        allTrainVectors += trainVectors
        allTrainLabels += trainLabels

        allTestVectors += testVectors
        allTestLabels  += testLabels

    return np.array(allTrainVectors, np.float32), np.array(allTrainLabels), np.array(allTestVectors, np.float32), np.array(allTestLabels)

def getTestPointsOnImg(img, regionSize):
    row, col = img.shape

    testPoints = []
    i, j = regionSize // 2, regionSize // 2

    # get all points within regionSize distance
    while i < row:
        testPoints.append([i, j])

        j += regionSize
        if j > col:
            j = regionSize // 2
            i += regionSize

    return testPoints

def showComparison(imgDir, svmModel, regionSize):
    directory = os.listdir(imgDir)

    for fileName in directory:
        path = os.path.join(imgDir, fileName)

        # read the image
        img = cv2.imread(path)
        imgCopy = img.copy()

        # apply gaussian blur and convert to grayscale image
        blurredImg = cv2.GaussianBlur(img, (5, 5), 0)
        imgGray = cv2.cvtColor(blurredImg, cv2.COLOR_RGB2GRAY)

        # get all points within regionSize distance
        # and calculate their feature vectors
        testPoints = getTestPointsOnImg(imgGray, regionSize)
        gradients = calculateFeatureVectors(imgGray, testPoints, regionSize)

        # predict the feature vectors
        testGradients = np.array(gradients, dtype = np.float32)
        predictions = svmModel.predict(testGradients)[1]

        # apply the harris corner detector too for later comparison
        dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
        cornerPointsByHarris = selectCornerPoints(dst, thresholdRate = 0.01)
        cornerLabelsOfHarris = createLabels(cornerPointsByHarris, [])

        # draw points on the images
        drawPoints(img, testPoints, predictions, drawOnlyCorners = True)
        drawPoints(imgCopy, cornerPointsByHarris, cornerLabelsOfHarris, drawOnlyCorners = True)

        # show the comparison
        plotComparison(img, imgCopy)

imgDir = "./img"

def main(argv):
    opts, args = getopt.getopt(argv, "t")

    isTesting = False
    regionSize = 5
    for opt, arg in opts:
        if opt == '-t':
            isTesting = Truetesting part

    if isTesting:
        # load the model
        svm = cv2.ml.SVM_load("svm_model_09")

        # create feature vectors on images, check their corner or noncorner state
        # show the predicted and harris corner points for comparison
        showComparison(imgDir, svm, regionSize)
    else:
        # get training, testing vectors and labels
        trainX, trainY, testX, testY = createTrainTestData(imgDir)

        # create and configure the model
        svm = cv2.ml.SVM_create()

        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

        # train the model
        svm.train(trainX, cv2.ml.ROW_SAMPLE, trainY)

        # save the model
        svm.save("svm_model")

        # make predictions
        predictions = svm.predict(testX)[1]

        cm = confusion_matrix(testY, predictions, labels = [1, -1])
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / len(testY)

        # show confusion matrix and overall accuracy
        print(cm)
        print(accuracy)

if __name__ == "__main__":
    main(sys.argv[1: ])