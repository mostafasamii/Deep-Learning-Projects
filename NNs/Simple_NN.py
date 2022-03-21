import numpy as np
from scipy import optimize

#Formulas we apply :
# Z(1) = X*W(1)
# a(2) = f(Z(1))
# Z(3) = a(2) * W(2)
# Yhat = f(Z(3))


# note :  X and y is 2 arrays
class NeuralNetwork(object):
    def __init__(self):
        #define HyperParameters
        self.inputLayersSize = 2
        self.outputlayersSize = 1
        self.hiddenLayersSize = 3

        #Initialize weigths:
        self.W1 = np.random.randn(self.inputLayersSize, self.hiddenLayersSize)
        self.W2 = np.random.randn(self.hiddenLayersSize, self.outputlayersSize)

        #Sigmoid function:
        def Sigmoid(z):
            #Apply sigmoid activation Function
            return 1/(1 + np.exp(z))

        #Sigmoid prime to compute yHat with respect to Z
        def SigmoidPrime(z):
            #derivate of sigmoid function
            return np.exp(-z)/((1 + np.exp(-z))**2)

        # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
        #def sigmoid(x):
        #    return math.tanh(x)

        #above Formulas
        def forwardpropagation(self, X):
            #propagate inputs through network
            self.z2 = np.dot(X, self.W1)
            self.a2 = self.Sigmoid(self.z2)
            self.z3 = np.dot(a2, self.W2)
            self.yHat = self.Sigmoid(z3)
            return yHat

         def costFunction(self, X, y):
             #Compute cost for given X,y, use weights already stored in class.
             self.yHat = self.forward(X)
             #We added a term to the cost function return to medicate the overftting using regularization
             #and then normalize the other part of the cost function to make sure that the ratio of the 2 error
             #doesnt change with respect to the number of the examples
             J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2) * (sum(self.W1**2) + sum(self.W2**2))
             return J

        #6th Formula
        def CostFunctionPrime(self, X, y):
            #compute derivate with respect to W1 and W2
            self.yHat = self.forwardpropagation(X)

            delta3 = np.multiply(-(y - self.yHat), self.SigmoidPrime(self.z3))
            #adding the gradient of the regularization term
            dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2

            delta2 =  np.dot(delta3, self.W2.T)*self.SigmoidPrime(self.z2)
            #adding the gradient of the regularization term
            dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1
            return dJdW1, dJdW2

        #helper functions for interacting with other methods or classes
        def getParams(self):
            #Get W1 and W2 rolled into a vector:
            param = np.concatenate((self.W1.ravel(), self.W2.ravel()))
            return param
        def setParam(self, params):
            #Set W1 and W2 using single parameter vector
            W1_start = 0
            W1_end = self.hiddenLayersSize * self.inputLayersSize
            self.W1 = np.reshape(param[W1_start : W1_end], self.inputLayersSize, self.hiddenLayersSize)
            W2_end = W1_end + (self.hiddenLayersSize*self.outputlayersSize)
            self.W2 = np.reshape(param[W1_end : W2_end], self.hiddenLayersSize, self.outputlayersSize)

        def computeGradients(self, X, y):
            dJdW1, dJdW2 = self.CostFunctionPrime(X, y):
                return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

        #****Numerical gradient checking****
        # at the end after we return and we want to know how far
        # we calculate one gradient at a time to make it simple
        #to divide the norm of the difference by the norm of the sum of the vectors we would like to compare.
        # Typical results should be on the order of 10^-8 or less if you’ve computed your gradient correctly.
        # norm(grad-numgrad)/norm(grad+numgrad)
        def computeNumericalGradients(N, X, y):
            paraminitial = N.getParams()
            numgrad = np.zeros(paraminitial.shape)
            perturb = np.zeroes(paraminitial.shape())
            e = 1e-4
            for p in range(len(paraminitial)):
                #Set perturbation vector
                perturb[p] = e
                N.setParam(paraminitial + perturb)
                loss2 = N.CostFunction(X, y)

                N.setParam(paraminitial - perturb)
                loss1 = N.CostFunction(X, y)

                #Then we calculate the slope between those values
                #computre numerical gradient
                numgrad[p] = (loss2 - loss1)/(2*e)
                #return the value we changed to zero
                perturb[p] = 0

            #return params to original value:
            N.setParam(paraminitial)

            return numgrad

#Using Quasi netwrok (BFSG) training Algorithm
#Works only with batch learning , which means if we can't use SGD with it
class training(object):
    def __init__(self, N):
        #Make local reference to NeuralNetwork
        self.N = N

    def costFunctionWrapper(self, param, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.computeGradients(X, y)
        return cost, grad

    #this function to track the cost function value as we train out network
    def callBackFunc(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        #make an internal variable for callback function:
        self.X = X
        self.y = y

        #make empty list to store costs
        self.J = []

        params0 = self.N.getParams()
        options = {'maxiter' : 200, 'disp' : True}
        #to use BFGS the minimize function requires to pass an objective function that accept a vector of parameters input and output data
        # and return both the cost and gradient, and the NN doesnt follow this simantics so we use Wrapper function to give it this behavior
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args = (X, y), options = options, callback = callBackFunc)

        #once the network is trained we replace the original parameters with the trained parameters
        self.N.setParams(_res.x)
        self.optimizationResults = _res
