import theano
import theano.tensor as T
import numpy as np

train_data = open('hw1/MLDS_HW1_RELEASE_v1/fbank/train.ark','r')
label_data = open('hw1/MLDS_HW1_RELEASE_v1/label/train.lab','r')
#phone_data = open('hw1/MLDS_HW1_RELEASE_v1/phones/48_39.map','r')

ans_data = open('answer_map.txt','w')
'''
test = phone_data.readline()
input_x = test.split()
print input_x
'''
train = T.matrix()
label = T.matrix()
ans   = T.matrix()
#phone = T.matrix()
'''
A=[]
B=[1,2]
C=[3,4]
A.append(B)
A.append(C)
print A
A.pop(1)
print A
'''
def float_convert(i):
	try: 
		return float(i)
	except ValueError :
		return i

#input_x = [float_convert(i) for i in input_x]
#print input_x
#print test

train = []
label_1 = []
label_2 = []
phone = []
ans = [] 
name_1=[]
name_2=[]
name_3=[]

for line in train_data:
	input_x = line.split()
	input_x = [float_convert(i) for i in input_x]
	train.append(input_x)
#print train[0][0]
#print len(train)

for line in label_data:
	label_x = line.split(',')
	name_x = label_x[0].split('_')

	label_1.append(label_x[0])
	label_2.append(label_x[1])
	name_1.append(name_x[0])
	name_2.append(name_x[1])
	name_3.append(name_x[2])

'''
for line in phone_data:
	phone_x = line.split()
	phone.append(phone_x)
'''

i=0
ini_find=0
#for i in range(0,len(train)-1) :
while i < len(train):
	if i%10000==0:
		print i
	if ini_find==0:
		row = label_1.index(train[i][0])
		name1_temp = name_1[row]
		name2_temp = name_2[row]
		ini_find=1
	else:
		ini_find = 1
		if row >= len(label_1):
			row = label_1.index(train[i][0])
			name1_temp = name_1[row]
			name2_temp = name_2[row]
		elif name_1[row]==name1_temp and name_2[row]==name2_temp:
			row = row
		else:
			row = label_1.index(train[i][0]) 
			name1_temp = name_1[row]
			name2_temp = name_2[row]
			
	ans.append(label_2[row])
	ans_data.write(label_1[row]+','+label_2[row])

	label_1.pop(row)
	label_2.pop(row)
	name_1.pop(row)
	name_2.pop(row)
	name_3.pop(row)
	
	i=i+1
	#row_phone = phone.index(label[row][1])
	#ans.append(phone[row_phone][1])

train_data.closed
label_data.closed
ans_data.closed


