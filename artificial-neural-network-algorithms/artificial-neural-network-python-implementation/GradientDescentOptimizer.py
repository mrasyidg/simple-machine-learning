from Optimizer import Optimizer


class GradientDescentOptimizer(Optimizer):
    def __init__(self, model, loss_func, learning_rate=1e-4):
        super().__init__(model, loss_func)
        self.learning_rate = learning_rate
    
    def update_layer(self,layer, grad_weights, grad_bias):
        layer.weights = layer.weights - self.learning_rate * grad_weights
        layer.bias = layer.bias - self.learning_rate * grad_bias