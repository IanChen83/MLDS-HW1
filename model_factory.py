import theano.tensor as T
import numpy as np
from theano import shared
__author__ = 'patrickchen'

'''
 Definitions and initialization parameters in this model
'''
neuron_number = 4       # Number of neurons in one layer
layer_number = 1        # Number of layers in this model
input_dimension = 64    # Dimension of input vector
output_dimension = 49   # Dimension of output vector
batch_number = 1        # Number of batch size


class ModelFactory:
    def __init__(self, _i_dim=64, _o_dim=49, _num_neuron=128, _num_layer=1, _num_batch=1, _lr=0.5):
        self.input_dim = _i_dim
        self.output_dim = _o_dim
        self.num_neuron = _num_neuron
        self.num_layer = _num_layer
        self.num_batch = _num_batch
        self.learning_rate = _lr
        pass

    def _create_shared(self):
        self.W = shared(T.fmatrix())
        pass

    def _act_function(self):
        pass

    def _layer_propagate(self, output, w, b, ):
        pass

    def set_learning_rate(self, _r):
        self.learning_rate = _r
