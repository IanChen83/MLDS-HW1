import theano.tensor as tensor
from theano import shared, function, grad
import numpy as np


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
        self.y_evaluated_function = None
        self.x_input = tensor.fmatrix("X_input")
        self.y_input = tensor.fmatrix("Y_input")
        self.W_array = []
        self.B_array = []

        # Deal with class initialization
        self._load_w_array()
        self._create_model()
        self._define_update_function()

    '''
        (Internal) Use layer_neuron_num_list to initialize W_array
    '''

    def _load_w_array(self):
        print "Load W array:\n\t%s\n\t(total %s hidden layer(s))" % (
            self.layer_neuron_num_list,
            len(self.layer_neuron_num_list)
        )
        temp = [self.input_dim] + self.layer_neuron_num_list + [self.output_dim]
        for i in range(len(temp) - 1):
            wp = np.random.uniform(-1, 1, (temp[i], temp[i + 1]))
            bp = np.random.uniform(-1, 1, (1, temp[i + 1]))

            # TODO: W and b are set to zeros
            self.W_array.append(shared(wp, name="W%d" % i, borrow=True))
            self.B_array.append(shared(bp, name="B%d" % i, borrow=True))

    def _create_model(self):
        result = self.x_input
        for i in range(len(self.W_array)):
            result = ModelFactory._layer_propagate(result, self.W_array[i], self.B_array[i])
        self.y_evaluated = result

    def _define_update_function(self):
        self.cost = ModelFactory._cost_function(self.y_evaluated, self.y_input)
        g = grad(self.cost, self.W_array + self.B_array)
        update_pairs = []
        j = len(self.W_array)
        for i in range(len(self.B_array)):
            update_pairs.append((self.W_array[i], self.W_array[i] - self.learning_rate * g[i]))
            update_pairs.append((self.B_array[i], self.B_array[i] - self.learning_rate * g[i + j]))

        self.update = function([self.x_input, self.y_input], g, updates=update_pairs,
                               allow_input_downcast=True)

    @staticmethod
    def _act_function(x):
        return (1 + tensor.tanh(x / 2)) / 2

    @staticmethod
    def _layer_propagate(layer_input, w, b):
        return ModelFactory._act_function(tensor.dot(layer_input, w) + b)

    @staticmethod
    def _cost_function(func, out):
        return (func - out).norm(1)

    def train_one(self, x_input, y_input):
        if len(x_input) != self.batch_num:
            print "Error: Input batch size not equals to the number pre-defined."
            exit(1)
        if len(x_input[0]) != self.input_dim or len(y_input[0]) != self.output_dim:
            print "Error: Input/Output dimension not equals to the number pre-defined."
            exit(1)
        return self.update(x_input, y_input)
