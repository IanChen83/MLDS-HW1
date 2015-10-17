#update_jason
import theano
import theano.tensor as T
import numpy as np
from numpy import linalg as LA

##########update()##############
x  = T.vector()
w1 = T.vector()
w2 = T.vector()
y_hat = T.vector()
LR = T.scalar() #learning rate
	# form b into w
a1 = x*w1
sigma1 = 1/(1+T.exp(a1))
a2 = w2*sigma1
sigma2 = 1/(1+T.exp(a2))
cost=(abs(sigma2-y_hat) ** 2).sum() ** (1. / 2)
g = T.grad(cost, [w1,w2])
w_new = [w1,w2] - LR*g
update = theano.function([x,w1,w2,y_hat,LR], w_new)
################################
print update([0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0,0,0],0.01)




'''
a=numpy.arange(9)-4
print a
print LA.norm(a)
b=(abs(a) ** 2).sum() ** (1. / 2)
print b


#theta_update = T.vector()
theta0_in = T.vector()
input_x_in = T.vector()
hiddle_a_in = T.vector()
learn_rate_in = T.scalar()
layer_in = T.scalar()
batch_in = T.scalar()
z_in = T.vector()
y_hat_in = T.vector()
#theta_update = T.vector()

def update(theta0,input_x,hiddle_a,learn_rate,layer,batch,z,y_hat):
	
	#y = T.vector()


	if layer==1:
		first_term = input_x
	else:
		first_term = hiddle_a
	y = 1/(1+T.exp(-1*z))
	d_segma = T.exp(-1*z)/(1+T.exp(-1*z))**2;
	#dC_y = T.grad( numpy.linalg.norm(y-y_hat) ,y-y_hat)
	k=(abs(y-y_hat) ** 2).sum() ** (1. / 2)
	dC_y = y-y_hat/k

	if layer==1:
		second_term = d_segma*dC_y
	else:
		second_term = 0
	#print first_term
	#print theta0
	print y
	#print second_term	
	#print d_segma
	#print dC_y

	theta_out = theta0 - learn_rate*first_term*second_term
	print theta_out
	
	y = 1/(1+T.exp(-1*z))
	k=(abs(input_x*theta0-y_hat) ** 2).sum() ** (1. / 2)
	theta_out = T.grad(k,theta0)
	
	return theta_out

theta0_in = np.array([0.1,0.1,0.1,0.1])
input_x_in = np.array([0.1, 0.2, 0.3, 1])
hiddle_a_in = np.array([0,0,0,0])
learn_rate_in = 1
layer_in = 1
batch_in = 1
z_in = np.array([0.1, 0.1, 0.1, 0.1])
y_hat_in = np.array([0, 1, 0, 1])

#print input_x_in*theta0_in
theta_update = update(theta0_in,input_x_in,hiddle_a_in,learn_rate_in,layer_in,batch_in,z_in,y_hat_in)
print theta_update

'''