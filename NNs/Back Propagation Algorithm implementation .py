#>>>>>>>>>>>>>>>>Stochastic gradient descent if needed <<<<<<<<<<
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
    #Train the neural network using mini-batch Stochastic gradient descent.
    #the training_data is a list of tuples (x,y) representing the training inputs
    #and the desired self-explanatory.
    #if the test_data is provided then the netwrok will be evalulated againts
    #the test data after each epochs and the partial progress printed out
    #this is useful for tracking progress but slow things down substantially
    if test_data : n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [
        training_data[k : k + mini_batch_size]
        for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            print("epochs{0} : {1} / {2}".format(j, self.evalulate(test_data), n_test))
        else:
            print("epoch{0} complete".format(j))
#**********************************************************************
def update_mini_batch(self, mini_batch, eta):
    #Update the network's weights and biases by applying gradient desecent
    #using a backpropagation to a single mini batch
    #the mini_batch is a list of tuples (x,y)  and eta the learning rate
    nabla_b = [np.zeroes(b.shape) for b in self.biases]
    nabla_w = [np.zeroes(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [np + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

def backprop(self, x, y):
    #return tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
    #nabla_b and nabla_w are layer-by-layer lists of numpy array
    #similar to self.biases and self.weights
    nabla_b = [np.zeroes(b.shape) for b in self.biases]
    nabla_w = [np.zeroes(w.shape) for w in self.weights]
    #feed forward:
    activation = x
    activations = [x] #list to store actications of all layers
    zs = [] #list to store all the z vectore layer by layers
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = self.non_linearity(z)
        activations.append(activation)
    #backward pass:
    delta = self.cost_derivative(activations[-1], y)* self.d_non_linearity(zs[-1])
    nabla_b[-1] = delta
    nabla_w [-1] = np.dot(delta, actications[-2], transpose())
    #note that the variable l in the loop below is used a little
    # l = 1 means the last layer of neurons , l = 2 is the second last layer
    for l in xrange(2, self.num_layers):
        z = zs[-l]
        sp = self.d_non_linearity(z)
        delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w [-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
