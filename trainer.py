from scipy import optimize

class Trainer(object):
    def __init__(self, NN):
        self.NN = NN

    def costFunctionWrapper(self, params, x, y):
        # Wrap NN
        self.NN.setParams(params)
        cost = self.NN.costFunction(x, y)
        grad = self.NN.computeGradients(x, y)
        return cost, grad # Returns [input data, output data]

    def callBackF(self, params):
        #Set minimize callback function
        self.NN.setParams(params)
        self.J.append(self.NN.costFunction(self.x, self.y))
        self.testJ.append(self.NN.costFunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX, testY):
        # Make internal variable for callback function
        self.x = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store costs
        self.J = []
        self.testJ = []

        params0 = self.NN.getParams() #Initial parameters

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, \
                                    jac=True, method='BFGS', args=(trainX, trainY), \
                                    options=options, callback=self.callBackF)
        # Uses BFGS method to calcute Gradient Descend
        self.NN.setParams(_res.x) # Replace original random params with trained params
        self.optimizationResults = _res
