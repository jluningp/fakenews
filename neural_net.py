import numpy as np
import random
from scipy.special import expit

class NeuralNet:
  def __init__(self, structure, learning_rate):
    """
    Initialize the Neural Network.

    - structure is a dictionary with the following keys defined:
        num_inputs
        num_outputs
        num_hidden
    - learning rate is a float that should be used as the learning
        rate coefficient in training

    When building your net, make sure to initialize your weights
    to random values in the range [-0.05, 0.05]. Specifically, you
    should use some transformation of 'np.random.rand(n,m).'
    """
    self.num_inputs = structure['num_inputs']
    self.num_outputs = structure['num_outputs']
    self.num_hidden = structure['num_hidden']
    self.learning_rate = learning_rate
    self.input_to_hidden = (np.random.rand(self.num_inputs+1, self.num_hidden) * 0.1 - 0.05) # 2D array of weights
    self.hidden_to_layer = (np.random.rand(self.num_hidden, self.num_hidden) * 0.1 - 0.05) # 2D array of weights
    self.layer_to_output = (np.random.rand(self.num_hidden, self.num_outputs) * 0.1 - 0.05) # 2D array of weights
    self.output = np.zeros(self.num_outputs)

  def get_weights(self):
    """
    Returns (w1, w2) where w1 is a matrix representing the current
    weights from the input to the hidden layer and w2 is a similar
    matrix for the hidden to output layers. Specifically, w1[i,j]
    should be the weight from input node i to hidden unit j.
    """
    return (self.input_to_hidden, self.layer_to_output)

  def forward_propagate(self, x):
    """
    Push the input 'x' through the network and returns the activations
    on the output nodes.

    - x is a numpy array representing an input to the NN

    Return a numpy array representing the activations of the output nodes.

    Hint: you may want to update state here, since you should call this
    method followed by back_propagate in your train method.
    """
    x = np.append(x, [1])
    self.prev_input = x

    self.hidden_output = expit(np.dot(x, self.input_to_hidden))
    self.layer_output = expit(np.dot(self.hidden_output, self.hidden_to_layer))

    self.output = expit(np.dot(self.layer_output, self.layer_to_output))

    return self.output
    

  def back_propagate(self, target):
    """
    Updates the weights of the NN for the last forward_propagate call.

    - target is the label of the last forward_propagate input
    """
    
    # find derivatives
    deriv_output = (target - self.output) * self.output * (1 - self.output)
    deriv_layer = self.layer_output * (1 - self.layer_output) * np.dot(self.layer_to_output, deriv_output)
    deriv_hidden = self.hidden_output * (1 - self.hidden_output) * np.dot(self.hidden_to_layer, deriv_layer)

    # update the weights
    self.layer_to_output = self.layer_to_output + (self.learning_rate * np.outer(self.layer_output, deriv_output))
    self.hidden_to_layer = self.hidden_to_layer + (self.learning_rate * np.outer(self.hidden_output, deriv_layer))
    self.input_to_hidden = self.input_to_hidden + (self.learning_rate * np.outer(self.prev_input, deriv_hidden))


  def train(self, X, Y, iterations=1000):
    """
    Trains the NN on observations X with labels Y.

    - X is a numpy matrix (array of arrays) corresponding to a series of
        observations. Each row is a new observation.
    - Y is a numpy matrix (array of arrays) corresponding to the labels
        of the observations.
    - iterations is how many passes over X should be completed.
    """
    for i in range(iterations):
        for row in range(len(X)):
            err = self.forward_propagate(X[row])
            if(err == Y[row]):
                self.learning_rate *= 0.9
            else:
                self.learning_rate *= 1.2
            self.back_propagate(Y[row])


        
  def test(self, X, Y):
    """
    Tests the NN on observations X with labels Y.

    - X is a numpy matrix (array of arrays) corresponding to a series of
        observations. Each row is a new observation.
    - Y is a numpy matrix (array of arrays) corresponding to the labels
        of the observations.

    Returns the mean squared error.
    """
    error = 0
    for row in range(len(X)):
        accuracy = (self.forward_propagate(X[row]) - Y[row])
        error += (np.dot(accuracy, accuracy)) * 1.0 / len(Y) * 1.0
    return error
