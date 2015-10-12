import theano

import theano.tensor
import numpy

from ModelFactory import *

print("-------- Start of Test 1 --------")
'''
    Test 1: grad of a matrix
'''
test1_x = tensor.fmatrix()
test1_x2 = tensor.fmatrix()
test1_x_sum = test1_x.norm(1) + test1_x2.norm(1)
test1_c = theano.function([test1_x, test1_x2], theano.grad(test1_x_sum, [test1_x, test1_x2]))
print(test1_c([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
print("--------- End of Test 1 ---------")

print("-------- Start of Test 2 --------")
'''
    Test 2: Use of shared variables
'''
test2_x = theano.shared(numpy.ones((3, 4), dtype='float32'), borrow=True)
test2_y = theano.tensor.fmatrix()
test2_1 = theano.dot(test2_x, test2_y)
test2_f = theano.function([test2_y], test2_1, updates=[(test2_x, test2_1)])
test2_f([[1],
         [1],
         [1],
         [1]])
print(test2_x.get_value())
print("--------- End of Test 2 ---------")

print("-------- Start of Test 3 --------")
'''
    Test 3: ModelFactory create_model
'''
"""
    * len(X)    = batch size
    * len(X[0]) = input dimension

    Notice that the dimension of the real X vector in this model is
        dim(input) + 1
    for elimination of b vector to simplify the computation.
"""
X = [[1, 2, 3]]

"""
    * len(W_number_list)    = Number of hidden layer
    * W_number_list[i]      = Number of neuron in layer[i]

    Notice that the dimension of the real W matrix is
        ( dim(output)+1, dim(input)+1 )
"""
W_number_list = [1]

layer_number = 1  # Number of layers in this model
input_dimension = len(X[0])  # Dimension of input vector
output_dimension = 1  # Dimension of output vector
batch_number = len(X)  # Number of batch size

test = ModelFactory(len(X[0]), 1, W_number_list, 1, 0.5)
print("--------- End of Test 3 ---------")
