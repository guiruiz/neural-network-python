import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def showCostPlot(J, testJ):
    plt.plot(J)
    plt.plot(testJ)
    plt.grid(1)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()

def showContourPlot(xx, yy, outputs):
    # Make Contour Plot
    CS = plt.contour(xx, yy, 100*outputs.reshape(100,100))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('Hours Sleep')
    plt.ylabel('Hours Study')
    plt.show()

def show3DPlot(xx, yy, outputs):
    # Make 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, 100*outputs.reshape(100,100), cmap=cm.jet)
    ax.set_xlabel('Hours Sleep')
    ax.set_ylabel('Hours Study')
    ax.set_zlabel('Test Scores')
    plt.show()

def showProjectionsPlot(x, y):
    # Make projections plot
    fig = plt.figure(0,(8,3))

    plt.subplot(1,2,1)
    plt.scatter(x[:,0],y)
    plt.grid(1)
    plt.xlabel('Hours Sleeping')
    plt.ylabel('Test Score')

    plt.subplot(1,2,2)
    plt.scatter(x[:,1], y)
    plt.grid(1)
    plt.xlabel('Hours Studying')
    plt.ylabel('Test Score')
    plt.show()

def computeNumericalGradient(nn, x, y):
    paramsInitial = nn.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4 #variation
    for p in range(len(paramsInitial)):
        perturb[p] = e # Set perturbation vector
        # Calc loss2
        nn.setParams(paramsInitial + perturb)
        loss2 = nn.costFunction(x, y)
        # Calc loss1
        nn.setParams(paramsInitial - perturb)
        loss1 = nn.costFunction(x, y)

        # Compute Numerical Gradient between
        numgrad[p] = (loss2 - loss1) / (2*e)
        # Return the valuye we changed back to zero
        perturb[p] = 0
    # Return params to original value
    nn.setParams(paramsInitial)
    return numgrad
