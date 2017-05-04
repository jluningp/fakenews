import numpy as np
import random
from scipy.special import expit

class NeuralNet:
  def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.num_hidden = num_hidden
    self.learning_rate = learning_rate
    self.input_to_hidden = (np.random.rand(self.num_inputs+1, self.num_hidden) * 0.1 - 0.05) # 2D array of weights
    self.hidden_to_layer = (np.random.rand(self.num_hidden, self.num_hidden) * 0.1 - 0.05) # 2D array of weights
    self.layer_to_output = (np.random.rand(self.num_hidden, self.num_outputs) * 0.1 - 0.05) # 2D array of weights
    self.output = np.zeros(self.num_outputs)

  def forward_propagate(self, x):
    x = np.append(x, [1])
    self.prev_input = x

    self.hidden_output = expit(np.dot(x, self.input_to_hidden))
    self.layer_output = expit(np.dot(self.hidden_output, self.hidden_to_layer))

    self.output = expit(np.dot(self.layer_output, self.layer_to_output))
    return self.output
    

  def back_propagate(self, target):
    # find derivatives
    deriv_output = (target - self.output) * self.output * (1 - self.output)
    deriv_layer = self.layer_output * (1 - self.layer_output) * np.dot(self.layer_to_output, deriv_output)
    deriv_hidden = self.hidden_output * (1 - self.hidden_output) * np.dot(self.hidden_to_layer, deriv_layer)

    # update the weights
    self.layer_to_output = self.layer_to_output + (self.learning_rate * np.outer(self.layer_output, deriv_output))
    self.hidden_to_layer = self.hidden_to_layer + (self.learning_rate * np.outer(self.hidden_output, deriv_layer))
    self.input_to_hidden = self.input_to_hidden + (self.learning_rate * np.outer(self.prev_input, deriv_hidden))


  def train(self, X, Y, iterations=1000):
    prev_error = 0
    threshold = 0.01
    for i in range(iterations):
      print("Running iteration {}".format(i))
      for row in range(len(X)):
            fp = self.forward_propagate(X[row])
            err = abs(fp - Y[row])
            if(err < prev_error and self.learning_rate > 0.001):
                self.learning_rate -= 0.001
            elif self.learning_rate < 0.5 and err > prev_error + threshold:
                self.learning_rate += 0.01
            prev_error = err
            self.back_propagate(Y[row])

        
  def test(self, X, Y):
    error = 0
    for row in range(len(X)):
        accuracy = (self.forward_propagate(X[row]) - Y[row])
        error += (np.dot(accuracy, accuracy)) * 1.0 / len(Y) * 1.0
    return error
