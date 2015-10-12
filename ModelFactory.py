import theano.tensor as tensor
import numpy as np
__author__ = 'patrickchen'


class ModelFactory:
    def __init__(self, _i_dim, _o_dim, _layer_neuron_num_list=None, _num_batch=1, _lr=0.5):
        # Deal with input parameter
        self.input_dim = _i_dim
        self.output_dim = _o_dim
        self.layer_neuron_num_list = _layer_neuron_num_list
        self.batch_num = _num_batch
        self.learning_rate = _lr
        self.result = 0
        self.W_array = []
        # Deal with class initialization
        self._load_w_array()

    def create_model(self, x):
        result = x
        for i in range(self.layer_num):
            result = ModelFactory._layer_propagate(result, self.W_array[i], self.b_array[i])
        self.result = result

    @staticmethod
    def _create_matrix_list(row, col):
        re = []
        for i in range(0, row):
            re.append([])
            for j in range(0, col):
                re[i].append(0)  # TODO: Use random number generator to get initial value
        return re

    @staticmethod
    def _act_function(x):
        return (1 + tensor.tanh(x / 2)) / 2

    @staticmethod
    def _layer_propagate(layer_input, w):
        return tensor.dot(layer_input, w)

    @staticmethod
    def _cost_function(func, out):
        return (func - out).norm(1)

    '''
        (Internal) Use layer_neuron_num_list to initialize W_array
    '''

    def _load_w_array(self):
        print "Load W array:\n\t%s\n\t(total %s hidden layers)" % (
            self.layer_neuron_num_list,
            len(self.layer_neuron_num_list)
        )
        temp = [self.input_dim] + self.layer_neuron_num_list + [self.output_dim]
        for i in range(1, len(temp) - 1):
            # TODO: W are set to zeros
            self.W_array.append(np.zeros((temp[i] + 1, temp[i + 1] + 1)))
        pass
