import theano.tensor as tensor

__author__ = 'patrickchen'


class ModelFactory:
    def __init__(self, _i_dim=64, _o_dim=49, _num_neuron=128, _num_layer=1, _num_batch=1, _lr=0.5):
        # Deal with input parameter
        self.input_dim = _i_dim
        self.output_dim = _o_dim
        self.num_neuron = _num_neuron
        self.layer_num = _num_layer
        self.batch_num = _num_batch
        self.learning_rate = _lr
        self.result = 0

        # Deal with class initialization
        #   W_array array
        self.W_array = []
        for _ in range(self.layer_num):
            self.W_array.append(tensor.fmatrix())
        # b_array array
        self.b_array = []
        for _ in range(self.layer_num):
            self.b_array.append(tensor.fvector())  # TODO: Use random number generator to get initial value

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
    def _layer_propagate(layer_input, w, b):
        _1 = tensor.dot(layer_input, w)
        _2 = _1 + b
        return ModelFactory._act_function(_2)

    @staticmethod
    def _cost_function(func, out):
        return (func - out).norm(1)
