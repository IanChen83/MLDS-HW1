import theano

from ModelFactory import *

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

'''
    Test 1: grad on a matrix
'''
x = tensor.fmatrix()
x_sum = x.norm(1)
c = theano.function([x], theano.grad(x_sum, x))
print(c([[1, 2], [3, 4]]))
