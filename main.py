import os
import time
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score

import sys, getopt

def plotImg(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plotComparison(predicted, harris):
    fontsize = "18"

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(predicted)
    ax[1].imshow(harris)

    ax[0].set_xlabel("SVM Corner Points", fontsize = fontsize)
    ax[1].set_xlabel("Harris Corner Points", fontsize = fontsize)

    plt.show()

def plotComparisonWithNoisy(predicted, predictedNoisy, harris, harrisNoisy, noiseAmount):
    fontsize = "18"

    fig, ax = plt.subplots(2, 2,figsize = (10, 8))

    ax[0][0].imshow(predicted)
    ax[0][1].imshow(harris)

    ax[1][0].imshow(predictedNoisy)
    ax[1][1].imshow(harrisNoisy)

    ax[0][0].set_xlabel("Predicted Corner Points", fontsize = fontsize)
    ax[0][1].set_xlabel("Harris Corner Points", fontsize = fontsize)

    ax[1][0].set_xlabel("Predicted Corner Points under Gaussian " + str(noiseAmount), fontsize = fontsize)
    ax[1][1].set_xlabel("Harris Corner Points under Gaussian " + str(noiseAmount), fontsize = fontsize)

    plt.show()

def drawPoints(img, points, labels, drawOnlyCorners = False):
    for index, point in enumerate(points):
        i, j = point
        isCorner = True if labels[index] == 1 else False

        if isCorner:
            cv2.circle(img,(j, i), 5, (255, 0, 0))
        elif not drawOnlyCorners:
            cv2.circle(img,(j, i), 5, (255, 255, 255))

def selectCornerPoints(harrisImg, thresholdRate):
    """
        Selects the corner points on the calculated harris image
        by choosing the points wrt maximum value of harris image * thresholdRate
    """

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
    """
        Selects the non corner points on the calculated harris image
        by checking the gradients of small regions (regionSize * regionSize)
        around random points.

        If the sum of the magnitude of the gradients are less than nonCornerThreshold
        the point is selected as non corner, expecting a flat region.
    """

    row, col = harrisImg.shape
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
        region = imgGray[i - (regionSize // 2) : i + (regionSize // 2) + 1, j - (regionSize // 2) : j + (regionSize // 2) + 1]
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
    """
        Selects corner and non corner points.

        Selects corner from the image applied harris
        Selects non corner from gray image by looking gradients
    """

    cornerPoints = selectCornerPoints(harrisImg, thresholdRate)
    nonCornerPoints = selectNonCornerPoints(harrisImg, imgGray, maxNumOfKeyPoints, regionSize)

    return cornerPoints, nonCornerPoints

def selectSamplePoints(points, lenSample, testRate = 0.1):
    """
        Selects arbitrary elements up to sample length
    """

    sampledPoints = random.sample(points, lenSample)
    sampleTrain, sampleTest = train_test_split(sampledPoints, test_size = testRate)

    return sampleTrain, sampleTest

def determineSampleLength(cornerPoints, nonCornerPoints):
    """
        Determines the sample length to create the dataset as
        equally divided for corner and non corner points.
    """

    lenCorner = len(cornerPoints)
    lenNonCorner = len(nonCornerPoints)

    lenSample = lenCorner
    if lenNonCorner < lenCorner:
        lenSample = lenNonCorner
    
    return lenSample

def createLabels(cornerPoints, nonCornerPoints):
    """
        Creates labels of the points.
        1 for corner points and -1 for non corner points
    """

    lenCorner = len(cornerPoints)
    lenNonCorner = len(nonCornerPoints)

    cornerTrainLabels = [1 for i in range(lenCorner)]
    nonCornerTrainLabels = [-1 for i in range(lenNonCorner)]

    labels = cornerTrainLabels + nonCornerTrainLabels
    return labels

def shuffleLists(pointsX, pointsY):
    """
        Shuffles the points with their labels
    """

    pointsZipList = list(zip(pointsX, pointsY))
    random.shuffle(pointsZipList)

    shuffledPointsX, shuffledPointsY = zip(*pointsZipList)
    return shuffledPointsX, shuffledPointsY

def getKeyPoints(imgGray, regionSize, shuffle = True):
    """
        Selects the training and testing points consisting of corner and non corner points
        for one image. These points are combined with other points from other images to create
        the dataset.
    """

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
    """
        Creates the feature vectors by calculating the gradients of the regions
        around the points and their magnitudes.

        Gets the region (regionSize x regionSize) around the point,
        applies Sobel operator to both x and y directions.
        Calculates the magnitude of these gradients.

        Flattens the calculated results and concatenates them to be a feature vector
        of size (regionSize x regionSize) x 3.
    """

    borderedImg = cv2.copyMakeBorder(img, regionSize // 2, regionSize // 2, regionSize // 2, regionSize // 2, cv2.BORDER_CONSTANT)

    featureVectors = []
    for point in points:
        i, j = point

        # shift the points to the true points, before the padding
        i, j = i + regionSize // 2, j + regionSize // 2

        region = borderedImg[i - (regionSize // 2) : i + (regionSize // 2) + 1, j - (regionSize // 2) : j + (regionSize // 2) + 1]

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

def createTrainTestData(imgDir, regionSize):
    """
        Traverses the given directory containing images.
        Gets the corner and non corner points on the images with training and testing sets.
        Calculates the corresponding feature vectors and creates their labels.

        Returns points and labels of the training and testing sets
    """

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
    """
        Traverses the image by regionSize * regionSize squares
        Returns the center point of the squares as the points to check
        their corner or noncorner state
    """

    row, col = img.shape

    testPoints = []
    i, j = regionSize // 2, regionSize // 2

    # get all points within regionSize distance
    while i < row - (regionSize // 2):
        testPoints.append([i, j])

        j += regionSize
        if j > col - (regionSize // 2):
            j = regionSize // 2
            i += regionSize

    return testPoints

def addGausianNoise(img, noiseAmount):
    """
        Applies a Gaussian noise to the given image
    """
    row, col, ch = img.shape

    normalizedImg = cv2.normalize(img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    noise = np.zeros((row, col, ch), dtype=np.float64)
    cv2.randn(noise, 0, noiseAmount)
    normalizedImg = normalizedImg + noise

    noisy = cv2.normalize(normalizedImg, None, 0.0, 255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return noisy

def applyMethods(img, regionSize, svmModel):
    """
        Applies SVM and Harris Corner Detectors on the given image
        Returns points, corresponding labels and
        the time elapsed for the calculations
    """

    # apply gaussian blur and convert to grayscale image
    blurredImg = cv2.GaussianBlur(img, (5, 5), 0)
    imgGray = cv2.cvtColor(blurredImg, cv2.COLOR_RGB2GRAY)

    # get corner points from SVM and Harris
    cornerPointsBySVM, cornerLabelsOfSVM, elapsedTimeSVM = cornerSVM(imgGray, regionSize, svmModel)
    cornerPointsByHarris, cornerLabelsOfHarris, elapsedTimeHarris = cornerHarris(imgGray)

    return (cornerPointsBySVM, cornerLabelsOfSVM, elapsedTimeSVM), (cornerPointsByHarris, cornerLabelsOfHarris, elapsedTimeHarris)

def calculateAntiNoiseCriterion(normalInfo, noisyInfo):
    """
        Calculates Anti Noise Criterion of the methods.
        Finds the intersection of the corner points detected for the
        normal and noisy image by a method (either SVM or Harris).

        Then calculates and returns the score as |set1 ∩ set2| / max(|set1|, |set2|)
        Higher the score, more robust the method to noises
    """

    points, labels, _ = normalInfo
    noisyPoints, noisyLabels, _ = noisyInfo

    # get corner points
    cornerPoints = [points[index] for index, label in enumerate(labels) if label == 1]
    noisyCornerPoints = [noisyPoints[index] for index, label in enumerate(noisyLabels) if label == 1]

    nt1 = map(tuple, cornerPoints)
    nt2 = map(tuple, noisyCornerPoints)

    st1 = set(nt1)
    st2 = set(nt2)

    # find the intersection
    intersection = st1.intersection(st2)
    intersectionLen = len(intersection)
    minPointsLen = max(len(cornerPoints), len(noisyCornerPoints))

    # calculate the score
    score = 0.0
    if len(cornerPoints) != 0 and len(noisyCornerPoints) != 0:
        score = (intersectionLen / minPointsLen) * 100

    return score

def showComparison(imgDir, svmModel, regionSize, plotFigures, plotNoisyFigures):
    totalTimeSVM = 0
    totalTimeHarris = 0

    totalAntiNoisyScoreSVM = 0
    totalAntiNoisyScoreHarris = 0
    counter = 0

    noiseAmount = 0.05

    directory = os.listdir(imgDir)

    for fileName in directory:
        path = os.path.join(imgDir, fileName)

        # read the image
        img = cv2.imread(path)
        noisyImg = addGausianNoise(img, noiseAmount)

        # copy the image for later use
        imgCopy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgCopy2 = imgCopy.copy()
        imgCopy3 = imgCopy.copy()
        imgCopy4 = imgCopy.copy()

        # apply the svm and harris on the image separately
        svmInfo, harrisInfo = applyMethods(img, regionSize, svmModel)

        # apply the svm and harris on noisy image
        noisySvmInfo, noisyHarrisInfo = applyMethods(noisyImg, regionSize, svmModel)

        # calculate anti noise score
        antiNoiseScoreSVM = calculateAntiNoiseCriterion(svmInfo, noisySvmInfo)
        antiNoiseScoreHarris = calculateAntiNoiseCriterion(harrisInfo, noisyHarrisInfo)

        totalAntiNoisyScoreSVM += antiNoiseScoreSVM
        totalAntiNoisyScoreHarris += antiNoiseScoreHarris

        print("Anti Noisy score of SVM : {}".format(antiNoiseScoreSVM))
        print("Anti Noisy score of Harris : {}".format(antiNoiseScoreHarris))

        # extract necessary info from tuples
        cornerPointsBySVM, cornerLabelsOfSVM, elapsedTimeSVM = svmInfo
        cornerPointsByHarris, cornerLabelsOfHarris, elapsedTimeHarris = harrisInfo

        noisyCornerPointsBySVM, noisyCornerLabelsOfSVM, _ = noisySvmInfo
        noisyCornerPointsByHarris, noisyCornerLabelsOfHarris, _ = noisyHarrisInfo        

        # add elapsed time to total for later calculation of avg
        totalTimeSVM += elapsedTimeSVM
        totalTimeHarris += elapsedTimeHarris
        counter += 1

        print("Calculation time of SVM : {}".format(elapsedTimeSVM))
        print("Calculation time of Harris : {}\n".format(elapsedTimeHarris))

        # plot the detection results of svm and harris if required
        if plotFigures:
            # draw points on the images
            drawPoints(imgCopy, cornerPointsBySVM, cornerLabelsOfSVM, drawOnlyCorners = True)
            drawPoints(imgCopy2, cornerPointsByHarris, cornerLabelsOfHarris, drawOnlyCorners = True)

            if plotNoisyFigures:
                drawPoints(imgCopy3, noisyCornerPointsBySVM, noisyCornerLabelsOfSVM, drawOnlyCorners = True)
                drawPoints(imgCopy4, noisyCornerPointsByHarris, noisyCornerLabelsOfHarris, drawOnlyCorners = True)

                plotComparisonWithNoisy(imgCopy, imgCopy3, imgCopy2, imgCopy4, noiseAmount)
            else:
                # show the comparison
                plotComparison(imgCopy, imgCopy2)

    avgTimeSVM = totalTimeSVM / counter
    avgTimeHarris = totalTimeHarris / counter

    avgAntiNoiseScoreSVM = totalAntiNoisyScoreSVM / counter
    avgAntiNoiseScoreHarris = totalAntiNoisyScoreHarris / counter

    print("Average calculation time of SVM : {}".format(avgTimeSVM))
    print("Average calculation time of Harris : {}\n".format(avgTimeHarris))

    print("Average anti noisy time of SVM : {}".format(avgAntiNoiseScoreSVM))
    print("Average anti noisy time of Harris : {}\n".format(avgAntiNoiseScoreHarris))

def cornerHarris(imgGray):
    """
        Applies the Harris Corner Detector on the given image
        Returns selected corner points with their label labels (full of 1).
    """

    # apply the harris corner detector too for later comparison
    t1 = time.time()
    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
    t2 = time.time()

    cornerPointsByHarris = selectCornerPoints(dst, thresholdRate = 0.01)
    cornerLabelsOfHarris = createLabels(cornerPointsByHarris, [])

    return cornerPointsByHarris, cornerLabelsOfHarris, round((t2 - t1), 3)

def cornerSVM(imgGray, regionSize, svmModel):
    """
        Gets the test points on the given image.
        Calculates the gradients of near regions
        Employs the trained SVM to check whether points are corner or not

        Returns all the points with corresponding labels stating corner or noncorner 
    """

    t1 = time.time()
    # get all points within regionSize distance
    # and calculate their feature vectors
    testPoints = getTestPointsOnImg(imgGray, regionSize)
    gradients = calculateFeatureVectors(imgGray, testPoints, regionSize)

    # predict using the feature vectors
    testGradients = np.array(gradients, dtype = np.float32)
    predictions = svmModel.predict(testGradients)[1]

    t2 = time.time()

    return testPoints, predictions, round((t2 - t1), 3)

imgDir = "./img"

def main(argv):
    opts, args = getopt.getopt(argv, "m:r:t")

    isTesting = False
    modelPath = None
    regionSize = -1
    for opt, arg in opts:
        if opt == '-t':
            isTesting = True
        elif opt == '-m':
            modelPath = arg
        elif opt == '-r':
            regionSize = int(arg)

    if modelPath is None:
        print("Model path is needed")
        sys.exit(-1)
    elif regionSize == -1:
        print("Region size is needed")
        sys.exit(-1)

    if isTesting:
        # load the model
        svm = cv2.ml.SVM_load(modelPath)

        # create feature vectors on images, check their corner or noncorner state
        # show the predicted and harris corner points for comparison
        showComparison(imgDir, svm, regionSize, plotFigures = True, plotNoisyFigures = False)
    else:
        # get training, testing vectors and labels
        trainX, trainY, testX, testY = createTrainTestData(imgDir, regionSize)

        # create and configure the model
        svm = cv2.ml.SVM_create()

        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

        # train the model
        svm.train(trainX, cv2.ml.ROW_SAMPLE, trainY)

        # save the model
        svm.save(modelPath)

        # make predictions
        predictions = svm.predict(testX)[1]

        kappa = cohen_kappa_score(testY, predictions)
        cm = confusion_matrix(testY, predictions, labels = [1, -1])
        accuracy = accuracy_score(testY, predictions)
        report = classification_report(testY, predictions)

        print(kappa)
        print(cm)
        print(accuracy)
        print(report)

        sn.set(font_scale=1.4) # for label size
        sn.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size
        plt.savefig(modelPath)

if __name__ == "__main__":
    main(sys.argv[1: ])