import numpy as np
from layers import Layer
class ReLU(Layer):
    def forward(self,input):
        self.input = input
        return np.maximum(0, input)

    def backward(self,grad_output):
        mask = self.input>0
        return grad_output * mask

class Sigmoid(Layer):
    def __init__(self):
        self.F_x = None
    
    def forward(self,input):
        self.input = input
        self.F_x = (1/(1+(np.exp(-input))))
        return self.F_x 
    def backward(self,grad_output):
        return grad_output * (self.F_x*(1-self.F_x))
    
class Tanh(Layer):
    def __init__(self):
        self.F_x = None

    def forward(self, input):
        self.input = input
        self.F_x = np.tanh(input)
        return self.F_x
    def backward(self, grad_output):
        return grad_output * (1 - (self.F_x ** 2))
    
class SoftMax(Layer):
    def forward(self, input):
        self.input = input
        shifted = input - np.max(input, axis=1, keepdims=True)
        return (np.exp(shifted)/np.sum(np.exp(shifted), axis=1, keepdims=True))
    def backward(self, grad_output):
        return grad_output
    