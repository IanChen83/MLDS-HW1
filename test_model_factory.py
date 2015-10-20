from ModelFactory import *


# ModelFactory.print_pic(test.grad, "/home/patrickchen/Desktop/a.png")
# ModelFactory.print_pic(test.grad, "/home/patrickchen/Desktop/cost.png")

# print test.grad(X, Y)
# print test.W_update[0].get_value()
# print test.grad_function[0](X, Y)
# print test.W_update[0].get_value()


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        pass


def my_print(title, content=None, switch=True):
    if switch is False:
        return
    if content is None:
        print BColors.OKGREEN + BColors.BOLD + BColors.UNDERLINE, "*", title, ":", BColors.ENDC
        return
    print BColors.OKGREEN + BColors.BOLD + BColors.UNDERLINE, "*", title, ":", BColors.ENDC, "\n", content


def my_assert(name, a, b, type='digit'):
    ret = False
    if type == 'digit':
        if np.linalg.norm(a - b) < 0.0001:
            ret = True
    elif type == 'string':
        if a == b:
            ret = True
    else:
        print BColors.FAIL, "assert '%s' type=%s failed, \n" % (name, type), "a =", a, "\nb =", b, BColors.ENDC
        return
    if ret:
        print BColors.OKBLUE + BColors.BOLD + BColors.UNDERLINE, "Assert '%s' succeed" % name + BColors.ENDC
    else:
        print BColors.FAIL + BColors.BOLD + BColors.UNDERLINE, "Assert '%s' failed" % name + BColors.ENDC


def test1(verbose=False):
    """
        Test 1: ModelFactory cost function and evaluate function
    """
    print("-------- Start of Test 1 --------")
    x = [[1.2, 1.0], [1.4, 1.0]]
    y = [[2.0, 1.1], [3.0, 1.0]]

    # print "* input x:\n\t", x
    # print "* input y:\n\t", y
    my_print("input x", x, verbose)
    my_print("input y", y, verbose)
    w_number_list = []

    input_dimension = len(x[0])  # Dimension of input vector
    output_dimension = len(y[0])  # Dimension of output vector
    batch_number = len(x)  # Number of batch size

    test = ModelFactory(input_dimension, output_dimension, w_number_list, batch_number, 0.01)
    my_print("W", test.W[0].get_value(), verbose)
    my_print("B", test.B[0].get_value(), verbose)

    _1 = np.dot(np.matrix(x), test.W[0].get_value())
    my_print("Step 1", _1, verbose)

    _2 = _1 + test.B[0].get_value()
    my_print("Step 2", _2, verbose)

    _3 = (1 + np.tanh(_2 / 2)) / 2
    my_print("Step 3,", _3, verbose)

    _4 = _3 - np.matrix(y)
    my_print("Step 4", _4, verbose)

    _5 = np.linalg.norm(_4, ord=2, axis=1)
    my_print("Step 5", _5, verbose)

    my_print("Evaluate function", test.y_evaluated_function(x), verbose)

    my_print("Cost function", test.cost_function(x, y), verbose)

    my_assert("Evaluate function", _3, test.y_evaluated_function(x))
    my_assert("Cost function", _5, test.cost_function(x, y))

    print("--------- End of Test 1 ---------")


def test2(verbose=False):
    """
        Test 2: ModelFactory grad function
        Base on Test 1
    """
    print("-------- Start of Test 2 --------")
    x = [[1.2, 0.9], [1.4, 0.4]]
    y = [[2.0, 1.1], [3.0, 0.7]]

    # print "* input x:\n\t", x
    # print "* input y:\n\t", y
    my_print("input x", x, verbose)
    my_print("input y", y, verbose)
    w_number_list = []

    input_dimension = len(x[0])  # Dimension of input vector
    output_dimension = len(y[0])  # Dimension of output vector
    batch_number = len(x)  # Number of batch size

    test = ModelFactory(input_dimension, output_dimension, w_number_list, batch_number, 0.01)

    my_print("W", test.W[0].get_value(), verbose)
    my_print("B", test.B[0].get_value(), verbose)

    a = np.dot(np.matrix(x), test.W[0].get_value()) + test.B[0].get_value()
    y_e = test.y_evaluated_function(x)
    cost = test.cost_function(x, y)

    my_print("Grad[0]", test.grad_function_no_update[0](x, y), verbose)
    my_print("Grad[1]", test.grad_function_no_update[1](x, y), verbose)

    grad_w_0 = np.zeros((2, 2))
    # print (y_e[0][0] * y_e[0][0]) * np.exp(-a[0].item((0, 0))) * x[0][0] / cost[0] * (y_e[0][0] - y[0][0])
    grad_w_0[0][0] = (y_e[0][0] * y_e[0][0]) * np.exp(-a.item((0, 0))) * x[0][0] / cost[0] * (y_e[0][0] - y[0][0])
    grad_w_0[0][1] = (y_e[0][1] * y_e[0][1]) * np.exp(-a.item((0, 1))) * x[0][0] / cost[0] * (y_e[0][1] - y[0][1])
    grad_w_0[1][0] = (y_e[0][0] * y_e[0][0]) * np.exp(-a.item((0, 0))) * x[0][1] / cost[0] * (y_e[0][0] - y[0][0])
    grad_w_0[1][1] = (y_e[0][1] * y_e[0][1]) * np.exp(-a.item((0, 1))) * x[0][1] / cost[0] * (y_e[0][1] - y[0][1])
    my_assert("Grad on W of batch 0", grad_w_0, test.grad_function_no_update[0](x, y)[0])

    grad_b_1 = np.zeros((2, 2))
    grad_b_1[0][0] = (y_e[1][0] * y_e[1][0]) * np.exp(-a.item((1, 0))) / cost[1] * (y_e[1][0] - y[1][0])
    grad_b_1[0][1] = (y_e[1][1] * y_e[1][1]) * np.exp(-a.item((1, 1))) / cost[1] * (y_e[1][1] - y[1][1])
    grad_b_1[1][0] = (y_e[1][0] * y_e[1][0]) * np.exp(-a.item((1, 0))) / cost[1] * (y_e[1][0] - y[1][0])
    grad_b_1[1][1] = (y_e[1][1] * y_e[1][1]) * np.exp(-a.item((1, 1))) / cost[1] * (y_e[1][1] - y[1][1])
    my_assert("Grad on B of batch 1", grad_b_1, test.grad_function_no_update[1](x, y)[1])

    print("--------- End of Test 2 ---------")


def test3(verbose=False):
    """
        Test 3: ModelFactory param update functionality
        Base on Test 1, 2
    """
    print("-------- Start of Test 3 --------")
    x = [[1.2, 0.9], [1.4, 0.4]]
    y = [[0.1, 1.1], [0.0, 0.7]]

    my_print("input x", x, verbose)
    my_print("input y", y, verbose)
    w_number_list = [2]

    input_dimension = len(x[0])  # Dimension of input vector
    output_dimension = len(y[0])  # Dimension of output vector
    batch_number = len(x)  # Number of batch size

    test = ModelFactory(input_dimension, output_dimension, w_number_list, batch_number, 0.01)

    print test.W_velocity[0].get_value()
    print test.B_velocity[0].get_value()

    _1w = test.W[0].get_value()
    _1b = test.B[0].get_value()
    my_print("W[0] (Before)", _1w, verbose)
    my_print("B[0] (Before)", _1b, verbose)

    _1 = []
    for i in range(test.batch_num):
        _1.append(test.grad_function[i](x, y))
        my_print("Grad on Batch %s" % i, _1[i], verbose)

    for i in range(test.layer_num):
        test.update_param_function[i](0)

    _2w = test.W[0].get_value()
    _2b = test.B[0].get_value()
    my_print("W[0] (After)", _2w, verbose)
    my_print("B[0] (After)", _2b, verbose)

    _3w = test.W_velocity[0].get_value()
    _3b = test.B_velocity[0].get_value()
    my_print("Velocity of W[0]", _3w, verbose)
    my_print("Velocity of B[0]", _3b, verbose)

    for i in range(test.layer_num):
        test.post_update_param_function[i](0)

    _4w = test.W_velocity[0].get_value()
    _4b = test.B_velocity[0].get_value()
    my_print("Velocity of W[0]", _4w, verbose)
    my_print("Velocity of B[0]", _4b, verbose)

    my_assert("Update W[0]", _2w, _1w + _3w)
    my_assert("Update B[0]", _2b, _1b + _3b)

    print("--------- End of Test 3 ---------")


def test4(verbose=False):
    """
        Test 4: ModelFactory Velocity update functionality
        Base on Test 1, 2, 3
    """
    print("-------- Start of Test 4 --------")
    x = [[1.2, 0.9], [1.4, 0.4]]
    y = [[0.1, 1.5], [0.5, 0.7]]

    my_print("input x", x, verbose)
    my_print("input y", y, verbose)
    w_number_list = []

    input_dimension = len(x[0])  # Dimension of input vector
    output_dimension = len(y[0])  # Dimension of output vector
    batch_number = len(x)  # Number of batch size

    test = ModelFactory(input_dimension, output_dimension, w_number_list, batch_number, 0.9)

    my_print("Momentum", test.update_momentum, verbose)

    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value(), verbose)
        my_print("B[%d]" % i, test.B[i].get_value(), verbose)

    test.train_one(x, y)

    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value(), verbose)
        my_print("B[%d]" % i, test.B[i].get_value(), verbose)

    print("--------- End of Test 4 ---------")


def test5(verbose=False):
    """
        Test 5: ModelFactory Convergence test
        Base on Test 1, 2, 3, 4
    """
    print("-------- Start of Test 5 --------")
    x = [[1.2, 0.9], [1.4, 0.4]]
    y = [[0.1, 0.5], [0.5, 0.2]]

    # x = [[1.2, 0.9]]
    # y = [[0.1, 0.5]]

    my_print("input x", x, verbose)
    my_print("input y", y, verbose)
    w_number_list = [3]

    input_dimension = len(x[0])  # Dimension of input vector
    output_dimension = len(y[0])  # Dimension of output vector
    batch_number = len(x)  # Number of batch size

    test = ModelFactory(input_dimension, output_dimension, w_number_list, batch_number, 0.001)
    for _ in range(5):
        test.train_one(x, y)

    my_print("(Before)")
    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value())
        my_print("W_Velocity[%d]" % i, test.W_velocity[i].get_value())

    for i in range(test.batch_num):
        my_print("Grad on batch[%d]" % i, test.grad_function[i](x, y))

    my_print("(After grad_function)")
    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value())
        my_print("W_Velocity[%d]" % i, test.W_velocity[i].get_value())

    for i in range(test.layer_num):
        test.update_param_function[i](0)

    my_print("(After update_param_function)")
    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value())
        my_print("W_Velocity[%d]" % i, test.W_velocity[i].get_value())

    for i in range(test.layer_num):
        test.post_update_param_function[i](0)

    my_print("(After post_update_param_function)")
    for i in range(test.layer_num):
        my_print("W[%d]" % i, test.W[i].get_value())
        my_print("W_Velocity[%d]" % i, test.W_velocity[i].get_value())

    print("--------- End of Test 5 ---------")


'''
 Use verbose to enable printing
 E.g. test1(True)
'''
# test1()
# test2()
# test3()
# test4()
test5(True)
