
import numpy as np

class NNlayer:

     def __init__(self, activation : callable, dimension : tuple [int, int], sample_space_weigths:  tuple[float, float], sample_space_bias :tuple[float, float]):
          self.activation = activation
          self.dimensions = dimension
          self.sample_space_weights = sample_space_weigths
          self.sample_space_bias = sample_space_bias
          self.weights = self._init_weights()
          self.bias = self._init_bias()
          
     
     def _init_weights(self):
          return np.random.uniform(self.sample_space_weights[0], self.sample_space_weights[1], (self.dimensions)).tolist()

     def _init_bias(self):
          return np.random.uniform(self.sample_space_bias[0], self.sample_space_bias[1], (self.dimensions)).tolist()
     
     #actually there is no need for bias activation since the identity function is applied there 
     def bias_activation(self, bias_before):
          pass

     def update_weights(self, local_output_error):
          # updates the weigths based on the local output layer of this specific layer times the 
          pass
     
     #biases will be updated with the identity function
     def update_bias(self, bias_error):
          pass

     
     def feed_forward(self, input) -> list[float]:
          # here numpy autamtically broadcasts our bias vector if we would start batching
          lin_comb = np.dot(np.array([self.weights]), np.array([input])) + np.array(self.bias)
          return self.activation(lin_comb)
     
     def __str__(self):
          return (f"ActivationFunctionObject("
                    f"activation={self.activation.__name__ if callable(self.activation) else str(self.activation)}, "
                    f"dimensions={self.dimensions}, "
                    f"sample_space_weights={self.sample_space_weights}, "
                    f"sample_space_bias={self.sample_space_bias}, "
                    f"weights={self.weights}, "
                    f"bias={self.bias})")
          




n = NNlayer(lambda x : x +1, 2, [-0.5, 0.5], [-0.5, 0.5] )
#print(n)

#print(n.feed_forward([2]))