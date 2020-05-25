import numpy as np

class Sigmoid(Layer):
    
    def __init__(self):
        pass

    def forward(self, _input):
        return(1/(1 + np.exp(-_input)))

    def backward(self, _input, grad_output):
        s = 1/(1+np.exp(-_input))
        sigmoid_backward = s *(1- s) # sf * (1 - sf)
        grad_input = grad_output * sigmoid_backward
        return [grad_input]

"""
You can use these statements to test it:
sigmoid = Sigmoid()
np.testing.assert_almost_equal(sigmoid.forward(np.array([[-99,-99], [0,0],[1,1], [99,99]])), np.array([[0, 0], [0.5,0.5], [0.7310586,0.7310586], [1, 1]]))
np.testing.assert_almost_equal(sigmoid.backward(np.array([[-99,-99], [99,99]]), np.ones((2,2)))[0], np.array([[0,0], [0, 0]]))
np.testing.assert_almost_equal(sigmoid.backward(np.array([[1, 1], [0,0]]), np.ones((2,2)))[0], np.array([[0.1966119, 0.1966119], [0.25, 0.25]]))
"""