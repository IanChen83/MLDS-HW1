import theano
import theano.tensor as tensor
import numpy


# Test 1: grad of a matrix
def test1():
    print("-------- Start of Test 1 --------")
    test1_x = tensor.fmatrix()
    test1_x2 = theano.shared(numpy.matrix([[1, 2]]), borrow=True)
    test1_x_sum = theano.dot(test1_x2, test1_x).norm(1)
    test1_c = theano.function([test1_x], theano.grad(test1_x_sum, [test1_x2]))
    print(test1_c([[1, 2], [3, 4]]))
    print("--------- End of Test 1 ---------")


# Test 2: Use of shared variables
def test2():
    print("-------- Start of Test 2 --------")
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


test1()
test2()
