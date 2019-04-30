import sys
import numpy as np
from nn import NeuralNetwork
from trainer import Trainer
import util

np.set_printoptions(formatter={'float': '{: 0.18f}'.format})

# Training Data:
trainX_orig = np.array([[3,5], [5,1], [10,2], [6,1.5]], dtype=float) # Input: [HoursSleep, HoursStudy]
trainY_orig = np.array([[75], [82], [93], [70]], dtype=float) # Output: [Test Score]

# Testing Data:
testX_orig = np.array([[4, 5.5], [4.5,1], [9,2.5], [6, 2]], dtype=float) # Input: [HoursSleep, HoursStudy]
testY_orig = np.array([[70], [89], [85], [75]], dtype=float) # Output: [Test Score]

#Nomalize data:
trainX = trainX_orig/np.amax(trainX_orig, axis=0)
trainY = trainY_orig/100 #Max test score is 100
#Nomalize data by max of training data:
testX = testX_orig/np.amax(trainX, axis=0)
testY = testY_orig/100 #Max test score is 100


NN = NeuralNetwork(0.0001)

# # Make sure our gradients are correct:
numgrad = util.computeNumericalGradient(NN, trainX, trainY)
grad = NN.computeGradients(trainX, trainY)
norm = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
print(norm) # Should be less than 1e-8:

T = Trainer(NN)
T.train(trainX, trainY, testX, testY)
# util.showCostPlot(T.J, T.testJ)

#Generate Test Data
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)
# Normalize data
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.
#Create 2-d versions of input for plotting
a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)
#Join into a single ionput matrix
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

# Calc values for graph ploting
yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1,100))).T
util.showContourPlot(xx, yy, allOutputs)
util.show3DPlot(xx, yy, allOutputs)
util.showProjectionsPlot(trainX_orig, trainY_orig)
