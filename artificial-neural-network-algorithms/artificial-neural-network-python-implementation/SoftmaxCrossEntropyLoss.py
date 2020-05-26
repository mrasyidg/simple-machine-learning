import numpy as np
from Softmax import Softmax

def softmax_cross_entropy(y_true, y_pred):
    softmax_cross_entropy_loss = - np.sum(y_true*np.log(y_pred))/y_true.shape[0]
    softmax_cross_entropy_grad = y_pred - y_true

    return softmax_cross_entropy_loss, softmax_cross_entropy_grad

def softmax_cross_entropy_with_logits(layer_input_output_cache, y_true, softmax=Softmax()):
    logits = layer_input_output_cache[-1]
    y_pred = softmax.forward(logits)

    return softmax_cross_entropy(y_true, y_pred)

"""
You can test it with these statements:

loss, _ = softmax_cross_entropy(np.array([[1,0],[0,1]]),np.array([[1-1e-9,0+1e-9],[1+1e-9,1-1e-9]]))
np.testing.assert_almost_equal(loss, 0)

loss, _ = softmax_cross_entropy(np.array([[1,0],[0,1],[0,1]]),np.array([[0.5,0.5],[0.5,0.5],[0.5,0.5]]))
np.testing.assert_almost_equal(loss, np.log(2))

loss, _ = softmax_cross_entropy(np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([[0.5,0.25,0.25],[0.5,0.25,0.25],[0.25,0.25,0.5]]))
np.testing.assert_almost_equal(loss, 0.9241962407465937)
"""
