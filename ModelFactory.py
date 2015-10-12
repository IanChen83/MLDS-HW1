import theano.tensor as tensor
from theano import shared, function, grad
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
        self.y_evaluated = None
        self.cost = None
        self.update = None
        self.x_input = tensor.fmatrix()
        self.y_input = tensor.fmatrix()
        self.W_array = []

        # Deal with class initialization
        self._load_w_array()
        self.create_model()
        self.define_update_function()

    def create_model(self):
        result = self.x_input
        for i in range(len(self.layer_neuron_num_list)):
            result = ModelFactory._layer_propagate(result, self.W_array[i])
        self.y_evaluated = result

    def define_update_function(self):
        self.cost = ModelFactory._cost_function(self.y_evaluated, self.y_input)
        g = grad(self.cost, self.W_array)
        update_pairs = []
        for i in range(len(self.W_array)):
            update_pairs.append((self.W_array[i], self.W_array[i] - self.learning_rate * g[i]))
        self.update = function([self.x_input, self.y_input], g, updates=update_pairs)

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
        return ModelFactory._act_function(tensor.dot(layer_input, w))

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
            self.W_array.append(
                shared(
                    np.zeros((temp[i] + 1, temp[i + 1] + 1), dtype='float32'),
                    borrow=None)
            )
        pass
