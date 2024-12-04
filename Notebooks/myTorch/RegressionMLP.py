
'''
This class buils up a Multilayer Perceptron from Scratch given arbitrary amount of Layers
'''

import numpy as np
from NNLayer import NNlayer

class RegressionMLP:

     def __init__(self, learning_rate : float , error_function : callable , stopping_threshold : float, max_runs : int, dimensions : list[int], activation : callable):
          self.dimensions_layers = dimensions
          self.activation = activation
          self.layers : list[NNlayer] = self._init_layers()
          # no need for output activation since it is the identity function
          self.learning_rate = learning_rate
          self.error_function = error_function
          self.stopping_treshold = stopping_threshold
          self.max_runs = max_runs
     
     def _init_layers(self):
          res = []
          for x in range(0, len(self.dimensions_layers) -1):
               res.append(NNlayer(self.activation, [self.dimensions_layers[x], self.dimensions_layers[x+1]], [-0.5, 0.5], [-0.5, 0.5]))
          return res
     
     # calculates the forward pass with matrix multiplication
     def forward_pass(self, input : float) -> float:
          layer_input = input
          for layer in self.layers:
               layer_input = layer.feed_forward(layer_input)
               print(layer)
          return layer_input
          
     def stochastic_gradient_descent(self):
          pass


r = RegressionMLP(0.5, lambda x: x+1, 0.1, 5, [1, 3, 1], np.tanh)
print(r.forward_pass(5))