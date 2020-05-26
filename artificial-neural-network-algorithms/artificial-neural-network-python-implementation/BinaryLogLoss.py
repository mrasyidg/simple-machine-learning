import numpy as np

def binary_log_loss(y_true, y_pred):
    """
        input: y_true, y_pred (numpy array with shape [batch_size, 1])
        output: bin_log_loss (numpy float64)
                bin_log_loss_grad (numpy array with shape [batch_size, 1])
    """
    N = y_pred.shape[0]
    bin_log_loss = -np.sum(np.multiply(np.log(y_pred),y_true) + np.multiply((1-y_true), np.log(1-y_pred)))/N
    bin_log_loss_grad = (y_pred-y_true)/((1-y_pred)*y_pred)
    return bin_log_loss, bin_log_loss_grad

def binary_log_loss_with_sigmoid(layer_input_output_cache, y_true):
    return binary_log_loss(layer_input_output_cache[-1], y_true)

"""
You can test it with these statements:

loss, grad = binary_log_loss(np.array([[1],[0]]),np.array([[1-1e-9],[0+1e-9]]))

np.testing.assert_almost_equal(loss, 0)
np.testing.assert_almost_equal(grad[0], -1)
np.testing.assert_almost_equal(grad[1], 1)

loss, grad = binary_log_loss(np.array([[1]]),np.array([[0.5]]))
np.testing.assert_almost_equal(loss, np.log(2))
assert grad[0][0] == -2
"""