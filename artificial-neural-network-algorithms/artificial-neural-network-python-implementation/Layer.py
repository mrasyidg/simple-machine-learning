import numpy as np

class Layer:

    def __init__(self):
        pass
    
    def forward(self, _input):
        return _input

    def backward(self, _input, grad_output):
        return [np.dot(grad_output, _input)]