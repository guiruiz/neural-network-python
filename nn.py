import numpy as np

class NeuralNetwork(object):

    def __init__(self, Lambda=0):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.Lambda = Lambda

    def forward(self, x):
        # Propagate inputs though network
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Sigmoid
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient (Derivative) of Sigmoid Function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, x, y):
        self.yHat = self.forward(x)
        #J = 0.5*np.sum((y-self.yHat)**2)  # CODE BEFORE FIX OVERFITTING
        J = 0.5*np.sum((y-self.yHat)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        #We don't want cost to increase with the number of examples, so normalize by dividing the error term by number of examples(X.shape[0])
        return J

    def costFunctionPrime(self, x, y):
        # Compute derivative with respect to W1 and W2
        self.yHat = self.forward(x)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3) )
        # dJdW2 = np.dot(self.a2.T, delta3) # CODE BEFORE FIX OVERFITTING
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/x.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        # dJdW1 = np.dot(x.T, delta2) # CODE BEFORE FIX OVERFITTING
        #Add gradient of regularization term:
        dJdW1 = np.dot(x.T, delta2)/x.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2

    # Helper functions for interacting with other methods/classes
    def getParams(self):
        # Get W1 and W2 rolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))

        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, x, y):
        dJdW1, dJdW2 = self.costFunctionPrime(x,y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
