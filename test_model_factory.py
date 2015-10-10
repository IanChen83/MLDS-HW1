from theano import tensor
import theano

from ModelFactory import *

X = [[1, 2, 3]]
B = [1]
W = [[1], [0], [0]]
Xnp = tensor.fmatrix()
neuron_number = len(W[0])  # Number of neurons in one layer
layer_number = 1  # Number of layers in this model
input_dimension = len(X[0])  # Dimension of input vector
output_dimension = 1  # Dimension of output vector
batch_number = len(X)  # Number of batch size

test = ModelFactory(input_dimension, output_dimension, neuron_number, 1, batch_number, 0.5)
test.create_model(Xnp)
f1 = theano.function([test.W_array] + [test.b_array] + [Xnp], test.result)

print f1(W, B, X)
