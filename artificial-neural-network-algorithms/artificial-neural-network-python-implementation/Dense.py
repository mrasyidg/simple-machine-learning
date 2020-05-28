import numpy as np
from Layer import Layer


class Dense(Layer):
    
    def __init__(self, input_shape, output_shape, weight_initializer='random'):

        if weight_initializer not in ['xavier','random']:
            raise ValueError('weight_initializer must be either xavier or random')

        if weight_initializer == 'random':
            self.weights = np.random.randn(input_shape, output_shape)*1e-2
        
        elif weight_initializer == 'xavier':
            self.weights = np.random.randn(input_shape, output_shape)*np.sqrt(2/(input_shape + output_shape))
        
        self.bias = np.random.randn(output_shape)
    
    def forward(self, _input):
        return np.dot(_input, self.weights) + self.bias
        
    def backward(self, _input, grad_output):
        grad_input = np.dot(grad_output, np.transpose(self.weights))
        grad_weights = np.dot(np.transpose(_input), grad_output)
        grad_bias = grad_output.mean(axis=0)*_input.shape[0]
        return [grad_input, grad_weights, grad_bias]


"""
You can test it with these statements:

np.random.seed(3)
dense = Dense(2,2)
np.testing.assert_almost_equal(dense.forward(np.ones([3,2])), np.array([[-0.25853694, -0.36902881], [-0.25853694, -0.36902881],[-0.25853694, -0.36902881]]))

dense.weights = np.ones((2,2))
dense.bias = np.zeros(2)
backprop = dense.backward(np.array([[1,1]]), np.array([[1,1]]))
np.testing.assert_almost_equal(backprop[0], np.array([[2., 2.]]))
np.testing.assert_almost_equal(backprop[1], np.array([[1., 1.], [1., 1.]]))
np.testing.assert_almost_equal(backprop[2], np.array([1., 1.]))

assert dense.weights.shape == backprop[1].shape
assert dense.weights.shape[0] == backprop[0].shape[1]
"""