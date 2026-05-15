import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer with Momentum.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, layers):
        """
        Loops through all layers in the network and updates their parameters
        (Weights and Biases) using the gradients calculated during backprop.
        """
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                # initialize its velocities to arrays of zeros.
                if layer not in self.velocities:
                    self.velocities[layer] = {
                        'dW': np.zeros_like(layer.W),
                        'db': np.zeros_like(layer.b)
                    }
                # Retrieve the current velocities for this layer
                v = self.velocities[layer]

                v['dW'] = self.momentum * v['dW'] + self.learning_rate * layer.dW
                layer.W = layer.W - v['dW']

                v['db'] = self.momentum * v['db'] + self.learning_rate * layer.db
                layer.b = layer.b - v['db']