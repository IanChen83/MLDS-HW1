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
        self.W_velocity = []
        self.B_velocity = []

        # training parameter
        self.x_input = tensor.fmatrix("X_input")
        self.y_input = tensor.fmatrix("Y_input")
        self.learning_rate = _lr

        self.cost_function = None

        self.y_evaluated = None
        self.y_evaluated_function = None

        self.grad = []
        self.grad_function = []
        self.grad_function_no_update = []

        self.random_mu = 0
        self.random_sigma = 0.1
        self.update_momentum = 0.5
        self.updates = []
        self.update_velocity = []
        self.update_param_function = []
        self.post_update_param_function = []

        '''
            Initialize and start
        '''
        self._initialize()
        self._create_model()
        self._create_param_updater()
        self._create_velocity_updater()
        self._create_post_velocity_updater()
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
            self.W_velocity.append(shared(wp_update, name="W_update%d" % i, borrow=True))

    '''
        (Internal) Initialize B array with random number
    '''

    def _initialize_b(self):
        temp = [self.input_dim] + self.hidden_layer_neuron_num_list + [self.output_dim]

        for i in range(len(temp) - 1):
            bp = np.random.normal(self.random_mu, self.random_sigma, (1, temp[i + 1]))
            bp_update = np.zeros((1, temp[i + 1]))

            self.B.append(
                shared(bp, name="B%d" % i, borrow=True, broadcastable=[True, False])
            )
            self.B_velocity.append(
                shared(bp_update, name="B_update%d" % i, borrow=True, broadcastable=[True, False])
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
        return ModelFactory._act_function(tensor.dot(layer_input, w) + b.dimshuffle('x', 1))

    @staticmethod
    def _cost_function(func, out):
        return (func - out).norm(2, axis=1)

    '''
    ####################### Update Model Creation #############################
    '''

    def _create_velocity_updater(self):
        for i in range(self.batch_num):
            self.grad.append([])
            self.grad[i] = tensor.grad(cost=ModelFactory._cost_function(self.y_evaluated, self.y_input)[i],
                                       wrt=self.W + self.B
                                       )

            # Define update pairs
            self.updates.append([])
            for j in range(self.layer_num):
                self.updates[i].append((
                    self.W_velocity[j],
                    self.W_velocity[j] - (self.learning_rate / self.batch_num) * self.grad[i][j]
                ))

            for j in range(self.layer_num):
                self.updates[i].append((
                    self.B_velocity[j],
                    self.B_velocity[j] -
                    (self.learning_rate / self.batch_num) * self.grad[i][j + self.layer_num]
                ))

            self.grad_function.append([])
            self.grad_function[i] = function([self.x_input, self.y_input],
                                             self.grad[i],
                                             allow_input_downcast=True,
                                             updates=self.updates[i]
                                             )
            self.grad_function_no_update.append([])
            self.grad_function_no_update[i] = function([self.x_input, self.y_input],
                                                       self.grad[i],
                                                       allow_input_downcast=True,
                                                       )

    def _create_param_updater(self):
        for i in range(self.layer_num):
            x = tensor.iscalar()
            updates_w = []
            updates_b = []
            updates_w.append((self.W[i], self.W[i] + self.W_velocity[i]))
            updates_b.append((self.B[i], self.B[i] + self.B_velocity[i]))
            _1 = function([x], outputs=x, updates=updates_b + updates_w)
            self.update_param_function.append(_1)

    def _create_post_velocity_updater(self):
        for i in range(self.layer_num):
            x = tensor.iscalar()
            updates_w = []
            updates_b = []
            updates_w.append((self.W_velocity[i], self.W_velocity[i] * self.update_momentum))
            updates_b.append((self.B_velocity[i], self.B_velocity[i] * self.update_momentum))
            _1 = function([x], outputs=x, updates=updates_b + updates_w)
            self.post_update_param_function.append(_1)
    '''
    ####################### Training Functions ################################
    '''

    def update(self, x, y):
        for i in range(self.batch_num):
            self.grad_function[i](x, y)

        for i in range(self.layer_num):
            self.update_param_function[i](0)

        for i in range(self.layer_num):
            self.post_update_param_function[i](0)

    def train_one(self, x_input, y_input):
        if len(x_input) != self.batch_num:
            print "Error: Input batch size not equals to the number pre-defined."
            exit(1)
        if len(x_input[0]) != self.input_dim or len(y_input[0]) != self.output_dim:
            print "Error: Input/Output dimension not equals to the number pre-defined."
            exit(1)
        self.update(x_input, y_input)
