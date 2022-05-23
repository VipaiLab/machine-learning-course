import sys
import os
import pickle
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('.\libsvm-master\python')
from svmutil import *
from numpy import *

filename = 'krkopt.data'
fr = open(filename)
arrayOLines = fr.readlines()
del arrayOLines[0]
numberOfLines = len(arrayOLines)
numberOfFeatureDimension = 6
data = zeros((numberOfLines, numberOfFeatureDimension))  # prepare matrix to return
label = zeros(numberOfLines)
for index in range(len(arrayOLines)):
    line = arrayOLines[index]
    listFromLine = line.split(',')
    data[index, 0] = ord(listFromLine[0])-96
    data[index, 1] = ord(listFromLine[1]) - 48
    data[index, 2] = ord(listFromLine[2])-96
    data[index, 3] = ord(listFromLine[3]) - 48
    data[index, 4] = ord(listFromLine[4])-96
    data[index, 5] = ord(listFromLine[5]) - 48
    if listFromLine[6] == 'draw\n':
        label[index] = 1
    else:
        label[index] = -1

permutatedData = zeros((numberOfLines, numberOfFeatureDimension))
permutatedLabel = zeros(numberOfLines)

p = random.permutation(numberOfLines)
for i in range(numberOfLines):
    permutatedData[i,:] = data[p[i],:]
    permutatedLabel[i] = label[p[i]]

numberOfTrainingData = 5000
xTrain = permutatedData[:numberOfTrainingData]
yTrain = permutatedLabel[:numberOfTrainingData]
xTest =  permutatedData[numberOfTrainingData:]
yTest = permutatedLabel[numberOfTrainingData:]

#subtract mean and divide by standard deviation
averageData = zeros((1, numberOfFeatureDimension))
for i in range(len(xTrain)):
    averageData += xTrain[i,:]

averageData = averageData/len(xTrain)

standardDeviation = zeros((1,numberOfFeatureDimension))

for i in range(len(xTrain)):
    standardDeviation+=(xTrain[i]-averageData[0,:])**2

standardDeviation = (standardDeviation/(len(xTrain)-1))**0.5

for i in range(len(xTrain)):
    xTrain[i] = (xTrain[i] -averageData)/standardDeviation

for i in range(len(xTest)):
    xTest[i] = (xTest[i] -averageData)/standardDeviation

#Search for good hyper-parameters. Firstly crude search.
CScale = [-5, -3, -1, 1, 3, 5,7,9,11,13,15]
gammaScale = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
maxRecognitionRate = 0
arr = np.array(xTrain)
newX = arr.tolist()
arr = np.array(yTrain)
newY = arr.tolist()
for i in range(len(CScale)):
    testC = 2 ** CScale[i]
    for j in range(len(gammaScale)):
        cmd = '-t 2 -c '
        cmd += str(testC)
        cmd += ' -g '
        testGamma = 2**gammaScale[j]
        cmd += str(testGamma)
        cmd += ' -v 5'
        cmd +=' -h 0'
        recognitionRate = svm_train(newY,newX, cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            print(maxRecognitionRate)
            maxCIndex = i
            maxGammaIndex = j

#Search for good hyper parameters. Second, refined search.
n = 10;
minCScale = 0.5*(CScale[max(0,maxCIndex-1)]+CScale[maxCIndex])
maxCScale = 0.5*(CScale[min(len(CScale)-1,maxCIndex+1)]+CScale[maxCIndex])
newCScale = arange(minCScale,maxCScale+0.001,(maxCScale-minCScale)/n)

minGammaScale = 0.5*(gammaScale[max(0,maxGammaIndex-1)]+gammaScale[maxGammaIndex])
maxGammaScale = 0.5*(gammaScale[min(len(gammaScale)-1,maxGammaIndex+1)]+gammaScale[maxGammaIndex])
newGammaScale = arange(minGammaScale,maxGammaScale+0.001,(maxGammaScale-minGammaScale)/n)

maxRecognitionRate = 0
for testCScale in newCScale:
    testC = 2 ** testCScale
    for testGammaScale in newGammaScale:
        testGamma = 2**testGammaScale
        cmd = '-t 2 -c '
        cmd += str(testC)
        cmd += ' -g '
        cmd += str(testGamma)
        cmd += ' -v 5'
        cmd +=' -h 0'
        recognitionRate = svm_train(newY,newX, cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            maxC = testC
            maxGamma = testGamma

#Input all training data to train again.
cmd = '-t 2 -c '
cmd += str(maxC)
cmd += ' -g '
cmd += str(maxGamma)
cmd += ' -h 0'
model = svm_train(newY,newX,cmd)

#Test
arr = np.array(xTest)
newX = arr.tolist()
arr = np.array(yTest)
newY = arr.tolist()
yPred, accuracy, decisionValues = svm_predict(newY, newX, model)
sio.savemat('yTest.mat', {'yTest': yTest})
sio.savemat('decisionValues.mat', {'decisionValues': decisionValues})

#drawROC
totalScores = sorted(decisionValues)
index = sorted(range(len(decisionValues)), key=decisionValues.__getitem__)
labels = zeros(len(yTest))
for i in range(len(labels)):
    labels[i] = yTest[index[i]]

truePositive = zeros(len(labels) + 1)
falsePositive = zeros(len(labels) + 1)
for i in range(len(totalScores)):
    if labels[i] > 0.5:
        truePositive[0] += 1
    else:
        falsePositive[0] += 1

for i in range(len(totalScores)):
    if labels[i] > 0.5:
        truePositive[i + 1] = truePositive[i] - 1
        falsePositive[i + 1] = falsePositive[i]
    else:
        falsePositive[i + 1] = falsePositive[i] - 1
        truePositive[i + 1] = truePositive[i]

truePositive = truePositive / truePositive[0]
falsePositive = falsePositive / falsePositive[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(falsePositive, truePositive)
plt.show()

