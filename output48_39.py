import theano
import theano.tensor as T
import numpy as np
__author__= 'jason'


class MAP:

    def __init__(self):
        self.map_data = open('/home/laplace/Desktop/dl/hw1/MLDS_HW1_RELEASE_v1/phones/48_39.map', 'r')
        self.speaker_data = open('/home/laplace/Desktop/dl/hw1/MLDS_HW1_RELEASE_v1/fbank/test.ark', 'r')
        self.in_48 = []
        self.in_39 = []
        self.speakerName = []	
        for line in self.map_data:
            in_x = line.split('\t')
            self.in_48.append(in_x[0])
            self.in_39.append(in_x[1])
        for line in self.speaker_data:
            self.speakerName.append(line.split()[0])

    def map(self, y):
        f = open('outputcsv.txt', 'w')
        f.write('Id,Prediction\n')
        for i in range(len(y)):
            big = max(y[i])
            big_index = y[i].index(big)
            if not i == len(y) - 1:
                f.write('%s,%s\n' % (self.speakerName[i], self.in_39[big_index].split('\n')[0]))
            else:
            	f.write('%s,%s' % (self.speakerName[i], self.in_39[big_index].split('\n')[0]))