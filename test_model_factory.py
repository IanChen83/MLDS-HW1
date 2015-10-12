import theano
import theano.tensor
import numpy

from ModelFactory import *

print("-------- Start of Test 1 --------")
'''
    Test 1: grad of a matrix
'''
test1_x = tensor.fmatrix()
test1_x2 = theano.shared(numpy.matrix([[1, 2]]), borrow=True)
test1_x_sum = theano.dot(test1_x2, test1_x).norm(1)
test1_c = theano.function([test1_x], theano.grad(test1_x_sum, [test1_x2]))
print(test1_c([[1, 2], [3, 4]]))
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
X = [[1.1, 0, 1.1]]
Y = [[0, 1.1, 0.5]]

"""
    * len(W_number_list)    = Number of hidden layer
    * W_number_list[i]      = Number of neuron in layer[i]

    Notice that the dimension of the real W matrix is
        ( dim(output)+1, dim(input)+1 )
"""
W_number_list = [128, 128]

input_dimension = len(X[0])  # Dimension of input vector
output_dimension = len(Y[0])  # Dimension of output vector
batch_number = len(X)  # Number of batch size

test = ModelFactory(input_dimension, output_dimension, W_number_list, batch_number, 0.5)
print test.W_array[0].get_value()
print test.B_array[0].get_value()

for i in range(10000):
    print "test %s" % i
    test.train_one(X, Y)
    # print test.y_evaluated
print test.y_evaluated_function(X, Y)
print("--------- End of Test 3 ---------")
