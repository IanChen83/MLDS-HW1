import theano.tensor as tensor
from theano import shared, function, printing
import numpy as np


class ModelFactory:
    def __init__(self, _i_dim, _o_dim, _hidden_layer_neuron_num_list=None, _num_batch=1, _lr=0.5):
        #  structural parameter
        self.input_dim = _i_dim
        self.output_dim = _o_dim
        self.hidden_layer_neuron_num_list = _hidden_layer_neuron_num_list
        self.layer_num = len(self.hidden_layer_neuron_num_list) + 1
        self.batch_num = _num_batch

        self.W = []
        self.B = []
        self.W_update = []
        self.B_update = []

        # training parameter
        self.x_input = tensor.fmatrix("X_input")
        self.y_input = tensor.fmatrix("Y_input")
        self.learning_rate = _lr

        self.cost_function = None

        self.y_evaluated = None
        self.y_evaluated_function = None

        self.grad = None
        self.grad_function = None

        self.random_mu = 0
        self.random_sigma = 0.1
        self.update_momentum = 0

        '''
            Initialize and start
        '''
        self._initialize()
        self._create_model()
        self._update_pairs()
        # self._define_velocity_function()

    '''
    ####################### Initialization ####################################
    '''
    '''
        Load W & B parameters explicitly
    '''

    def load_array(self):
        pass

    '''
        (Internal) Initialize parameters
    '''

    def _initialize(self):
        if len(self.W) == 0:
            self._initialize_w()
            self._initialize_b()

    '''
        (Internal) Initialize W array with random number
    '''

    def _initialize_w(self):
        temp = [self.input_dim] + self.hidden_layer_neuron_num_list + [self.output_dim]

        for i in range(len(temp) - 1):
            wp = np.random.normal(self.random_mu, self.random_sigma, (temp[i], temp[i + 1]))
            wp_update = np.zeros((temp[i], temp[i + 1]))

            self.W.append(shared(wp, name="W%d" % i, borrow=True))
            self.W_update.append(shared(wp_update, name="W_update%d" % i, borrow=True))

    '''
        (Internal) Initialize B array with random number
    '''

    def _initialize_b(self):
        temp = [self.input_dim] + self.hidden_layer_neuron_num_list + [self.output_dim]

        for i in range(len(temp) - 1):
            bp = np.random.normal(self.random_mu, self.random_sigma, (1, temp[i + 1]))
            bp_update = np.zeros((1, temp[i + 1]))

            self.B.append(
                shared(np.tile(bp, (self.batch_num, 1)), name="B%d" % i, borrow=True)
            )
            self.B_update.append(
                shared(bp_update, name="B_update%d" % i, borrow=True)
            )

    '''
    ####################### Model Creation ####################################
    '''
    '''
        (Internal) Create model
    '''
    def _create_model(self):
        y_evaluated = self.x_input
        for i in range(self.layer_num):
            y_evaluated = ModelFactory._layer_propagate(y_evaluated, self.W[i], self.B[i])

        self.y_evaluated = y_evaluated

        # No need to update the parameters to evaluate y
        self.y_evaluated_function = function([self.x_input],
                                             y_evaluated,
                                             allow_input_downcast=True,
                                             name="y_evaluated_function"
                                             )

        # Calculate cost function for debugging
        self.cost_function = function([self.x_input, self.y_input],
                                      (y_evaluated - self.y_input).norm(2, axis=1),
                                      allow_input_downcast=True,
                                      name="cost_function"
                                      )

    @staticmethod
    def print_pic(k, out):
        printing.pydotprint(k, outfile=out, var_with_name_simple=True)

    @staticmethod
    def _act_function(x):
        return (1 + tensor.tanh(x / 2)) / 2

    @staticmethod
    def _layer_propagate(layer_input, w, b):
        return ModelFactory._act_function(tensor.dot(layer_input, w) + b)

    @staticmethod
    def _cost_function(func, out):
        return (func - out).norm(2, axis=1)

    '''
    ####################### Update Model Creation #############################
    '''

    def reset_update(self):
        for i in self.B_update:
            i.set_value(self.update_momentum * i.get_value())
        for i in self.W_update:
            i.set_value(self.update_momentum * i.get_value())

    def _update_pairs(self):
        update_pairs_param = []
        update_pairs_Velocity = []
        self.reset_update()
        # self.grad[i] stands for the gradient of each input in a batch
        _grad = []
        _temp = ModelFactory._cost_function(self.y_evaluated, self.y_input)

        self.grad = []
        self.grad_function = []
        for i in range(self.batch_num):
            self.grad.append(tensor.grad(_temp[i], self.W + self.B))

        for i in range(self.batch_num):
            update_pairs_param.append([])

            for j in range(self.layer_num):
                update_pairs_param[i].append(
                    (self.W_update[j],
                     self.W_update[j] -
                     self.learning_rate * self.grad[i][j] / self.batch_num
                     )
                )

                update_pairs_param[i].append(
                    (self.B_update[j],
                     self.B_update[j] -
                     self.learning_rate * self.grad[i][j + self.layer_num][0] / self.batch_num
                     )
                )

            self.grad_function.append(function([self.x_input, self.y_input],
                                               self.grad[i],
                                               allow_input_downcast=True,
                                               name="grad",
                                               updates=update_pairs_param[i]
                                               )
                                      )
            #
            # for i in range(self.layer_num):
            #     update_pairs.append((self.W_array[i], self.W_array[i] - self.learning_rate * self.grad[i]))
            #     update_pairs.append((self.B_array[i], self.B_array[i] - ))

    #     self.cost = ModelFactory._cost_function(self.y_evaluated, self.y_input)
    #     self.g = grad(self.cost, self.W_array + self.B_array)
    #     update_pairs = []
    #     j = len(self.W_array)
    #     for i in range(len(self.B_array)):
    #         # Velocity calculation
    #         update_pairs.append((self.vB_array[i],
    #                              self.update_momentum * self.vB_array[i] - self.learning_rate * self.g[i + j] /
    #                              self.batch_num))
    #         update_pairs.append((self.vW_array[i],
    #                              self.update_momentum * self.vW_array[i] - self.learning_rate * self.g[i] /
    #                              self.batch_num))
    #
    #     for i in range(len(self.B_array)):
    #         # Parameter calculation
    #         update_pairs.append((self.W_array[i], self.W_array[i] + self.vW_array[i]))
    #         update_pairs.append((self.B_array[i], self.B_array[i] + self.vB_array[i]))
    #
    #     self.y_evaluated_function = function([self.x_input, self.y_input], self.y_evaluated,
    #                                          allow_input_downcast=True, on_unused_input='ignore')
    #     self.update = function([self.x_input, self.y_input], self.g, updates=update_pairs,
    #                            allow_input_downcast=True)
    #     # print dir(self.y_evaluated)
    #     pass
    #
    # def _define_velocity_function(self):
    #     pass
    #

    '''
    ####################### Training Functions ################################
    '''

    def update(self, x, y):
        for i in range(self.batch_num):
            self.grad_function[i](x, y)

        for i in range(self.layer_num):
            self.W[i].set_value(self.W[i].get_value() + self.W_update[i].get_value())
            self.B[i].set_value(self.B[i].get_value() + np.tile(self.B_update[i].get_value(), (self.batch_num, 1)))

    def train_one(self, x_input, y_input):
        if len(x_input) != self.batch_num:
            print "Error: Input batch size not equals to the number pre-defined."
            exit(1)
        if len(x_input[0]) != self.input_dim or len(y_input[0]) != self.output_dim:
            print "Error: Input/Output dimension not equals to the number pre-defined."
            exit(1)
        self.update(x_input, y_input)
