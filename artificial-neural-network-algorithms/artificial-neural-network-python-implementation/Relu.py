import numpy as np
from Layer import Layer


class Relu(Layer):
    def __init__(self):
        pass
    
    def forward(self, _input):
        return np.maximum(0, _input)
    
    def backward(self, _input, grad_output):
        relu_grad = _input > 0
        return [grad_output*relu_grad]