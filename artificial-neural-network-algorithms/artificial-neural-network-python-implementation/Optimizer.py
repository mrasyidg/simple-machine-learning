import numpy as np
import time
from tqdm import tqdm
import traceback


class Optimizer:
    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func
    
    
    def update_layer(self, layer, weight, bias):
        raise NotImplementedError

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
        
    def fit(self, X_train, y_train ,X_val=None, y_val=None,n_epoch=100, batch_size=None):

        if batch_size is None:
            batch_size = X_train.shape[0]

        total_m_train= X_train.shape[0]
       
        use_val = X_val is not None
        if use_val:
            total_m_val = X_val.shape[0]

        history_train = []
        history_val = []

        for epoch in range(n_epoch):

            p_bar = tqdm(total=int(total_m_train/batch_size) + 1 if total_m_train%batch_size != 0 else 0)
            try:
                p_bar.set_description('epoch {}: {}/{} loss: inf'.format(epoch+1, 0, total_m_train))

                history_per_batch = []
                for i in (range(0, total_m_train, batch_size)):

                    #separate each set to mini-batches
                    adjusted_batch_size = min(i+batch_size, total_m_train)
                    X_batch_train = X_train[i:adjusted_batch_size]
                    y_batch_train = y_train[i:adjusted_batch_size]

                    # do forward propagation
                    layer_input_output_cache = self._forward(X_batch_train)

                    # compute its loss and grad
                    loss, output_grad = self.loss_func(layer_input_output_cache, y_batch_train)
                    # report current progress
                    history_per_batch.append(loss)

                    p_bar.set_description('epoch {}: {}/{} loss: {:4f}'.format(epoch+1, adjusted_batch_size, total_m_train, loss))
                    p_bar.update(1)
                    
                    # do backward propagation
                    self._backward(layer_input_output_cache, output_grad)
                history_train.append(history_per_batch)

                train_loss_mean = np.mean(history_per_batch)
                val_loss_mean = None

                if use_val:
                    history_per_batch = []
                    for i in (range(0, total_m_val, batch_size)):

                        # separate each set into mini-batches
                        adjusted_batch_size = min(i+batch_size, total_m_val)
                        X_batch_val = X_val[i:adjusted_batch_size]
                        y_batch_val = y_val[i:adjusted_batch_size]

                        # do forward propagation
                        layer_input_output_cache = self._forward(X_batch_val)
                        loss, output_grad = self.loss_func(layer_input_output_cache, y_batch_val)

                        history_per_batch.append(loss)
                    history_val.append(history_per_batch)

                    val_loss_mean = np.mean(history_per_batch)
                val_report = ''
                if val_loss_mean is not None:
                    val_report = 'val: {:4f}'.format(val_loss_mean)
                p_bar.close()
                time.sleep(0.2)
                print('\tmean epoch {} loss: train: {:4f}'.format(epoch+1, train_loss_mean),val_report)
                time.sleep(0.2)
            except Exception as e:
                p_bar.close()
                traceback.print_stack()
                raise Exception(e)
                

        return history_train, history_val