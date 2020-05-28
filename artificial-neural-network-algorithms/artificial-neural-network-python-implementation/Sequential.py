from Layer import Layer


class Sequential:
    def __init__(self):
        self.layers = []
    
    def append(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)
    
    def predict_proba(self, X):
        _input = X
        for layer in self.layers:
            _input = layer.forward(_input)
        return _input