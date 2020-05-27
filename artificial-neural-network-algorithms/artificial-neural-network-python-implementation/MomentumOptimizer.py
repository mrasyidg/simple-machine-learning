class MomentumOptimizer(Optimizer):
    def __init__(self, model, loss_func, learning_rate=1e-4, beta=0.9): # with learning rate 0.0001
        super().__init__(model, loss_func)
        self.learning_rate = learning_rate
        self.beta = beta
        self.__reset_velocity()
    
    def __reset_velocity(self):
        self.velocity = {}
    
    def update_layer(self, layer, grad_weights, grad_bias):
        # initialize velocity to zero
        if id(layer) not in self.velocity:
            self.velocity[id(layer)] = 0
        
        self.velocity[id(layer)] = (self.beta * self.velocity[id(layer)] + ((1-self.beta) * grad_weights))
        layer.weights = (layer.weights - self.learning_rate * self.velocity[id(layer)])
        
        layer.bias = layer.bias - self.learning_rate * grad_bias