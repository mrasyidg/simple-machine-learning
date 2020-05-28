import numpy as np
from Layer import Layer


class Softmax(Layer):
    
    def __init__(self):
        pass

    def forward(self, _input):
        softmax = lambda x: np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
        return[softmax(i) for i in _input]

    def backward(self, _input, grad_output):
        return [grad_output]

"""
You can test it with these statements:
softmax = Softmax()
np.testing.assert_almost_equal(softmax.forward(np.array([[100],[1],[0]])), np.array([[1],[1],[1]]))
np.testing.assert_almost_equal(softmax.forward(np.array([[100,3000],[1,1],[0,1]])), np.array([[0,1],[0.5,0.5],[0.2689414, 0.7310586]]))
"""