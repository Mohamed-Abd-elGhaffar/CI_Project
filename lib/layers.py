import numpy as np

# =============================================================================
# BASE LAYER
# =============================================================================

class Layer:
    """
    Abstract base class for all layers.
    Every layer in our library must implement forward() and backward().

    """

    def forward(self, input):
        """
        Compute the layer's output given an input.
        Must be implemented by every subclass.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Given dL/dOutput (gradient of loss w.r.t. this layer's output),
        compute and return dL/dInput (gradient w.r.t. this layer's input).
        
        For layers with learnable parameters (like Dense), also compute
        and store dL/dW and dL/db here.
        """
        raise NotImplementedError


# =============================================================================
# DENSE (FULLY CONNECTED) LAYER
# =============================================================================

class Dense(Layer):
    """
    A fully connected layer: output = input @ W + b

    Each neuron in this layer is connected to every input feature.
    W shape: (input_size, output_size)
    b shape: (1, output_size)  — broadcast across the batch

    """

    def __init__(self, input_size, output_size, weight_init='he'):
        if weight_init == 'he':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == 'xavier':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            raise ValueError(f"unknown init: {weight_init}")
        
        self.b = np.zeros((1, output_size))

        # Gradients — will be computed during backward pass
        self.dW = None
        self.db = None

        # Cache
        self.input = None

    def forward(self, input):
        """
        Forward pass: apply the linear transformation.

        input shape:  (batch_size, input_size)
        output shape: (batch_size, output_size)

        """
        self.input = input  # Cache for backward pass
        return input @ self.W + self.b

    def backward(self, grad_output):
        """
        Backward pass: compute gradients using the chain rule.

        grad_output: dL/dOutput, shape (batch_size, output_size)
                     — received from the layer ahead (closer to the loss)

        """
        self.dW = self.input.T @ grad_output           # (input_size, output_size)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # (1, output_size)

        grad_input = grad_output @ self.W.T            # (batch_size, input_size)
        return grad_input
    
    
class Dropout(Layer):
    """
    Dropout layer: randomly sets input units to 0 with a frequency of `rate` 
    at each step during training time, which helps prevent overfitting.
    """
    def __init__(self, rate=0.2):
        self.rate = rate
        self.mask = None
        self.is_training = True # Dropout only happens during training!

    def forward(self, input):
        if not self.is_training:
            return input # Do nothing during testing/evaluation
            
        # Create a mask of 1s and 0s. 
        # np.random.binomial draws samples from a binomial distribution.
        self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)
        
        # Apply the mask and scale up the survivors
        return (input * self.mask) / (1 - self.rate)

    def backward(self, grad_output):
        return (grad_output * self.mask) / (1 - self.rate)