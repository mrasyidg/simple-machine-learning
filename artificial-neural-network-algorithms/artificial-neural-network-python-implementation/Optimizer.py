class Optimizer:
    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func
    
    
    # def update_layer(self, layer, weight, bias):
        

    # forward propagation   
    def _forward(self, _input):
        # cache the input and output to be used later
        layer_input_output_cache = [_input]   
        
        # forward propagation for each layer
        for layer in self.model.layers:
            _input = layer.forward(_input)
            layer_input_output_cache.append(_input)
        
        return layer_input_output_cache
    
    def _backward(self, layer_input_output_cache, output_grad):
        # reverse loop
        for j in range(len(self.model.layers))[::-1]:

            layer = self.model.layers[j]

            # list of gradient
            D_list = layer.backward(layer_input_output_cache[j], output_grad)

            # update output grad and
            if len(D_list) > 1:
                # if the list contains weight and bias
                # then update this layer's bias and weight
                [output_grad, weight, bias] = D_list
                self.update_layer(layer, weight, bias)
            else:
                output_grad = D_list[0]