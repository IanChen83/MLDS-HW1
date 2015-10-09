import random
import theano
import theano.tensor as T
class init_weight:
    weights = [[]]
    def __init__(self,prev_num,cur_num):
        self.prev_num = prev_num
        self.cur_num = cur_num
    def random(self):
        weights = [[random.random() for x in range(self.prev_num)] for x in range(self.cur_num)] 
if __name__ == '__main__':
    inputs = [0.1,0.5,0.32,0.45,0.67,0.91,0.33]
    layers_num = input("please input the number of layers:")
    layers_num = int(layers_num)
    neuron_num = [len(inputs)]

    for i in range (layers_num ):
        x = input("please input the number of neuron in layer:")
        x = int(x)
        neuron_num.append(x)
    a = T.matrix()
    b = T.matrix()
    c = a * b
    multiply = theano.function([a,b],c)    
    ni = inputs
    for i in range (layers_num - 1):
        w = init_weight(neuron_num[i+1],neuron_num[i])
        w.random()
        ni = multiply(w.weights,ni)
    print ni

    
