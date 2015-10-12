import os

def mycreate(feature="./resource/fbank/train.ark", answer="./resource/label/train.lab"):
    f = open(feature)
    speaker = list()
    value = list()
    value_list = list()
    ans = list()
    anstype = ["aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d", "dh", "dx", "eh", "el"
               , "en", "epi", "er", "ey", "f", "g", "hh", "ih", "ix", "iy", "jh", "k", "l", "m", "ng"
               , "n", "ow", "oy", "p", "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w"
               , "y", "zh", "z"]
    process = 0
    for line in open(feature):
        token = f.readline().split()
        speaker.append(token[0])
        for i in range(len(token)-1):
            value.append(float(token[i+1]))
        value_list.append(value)
        value=[]
        process = process + 1
        if process % 100 == 0:
            print("feature processing to line:  " + str(process))
    f.close()
    process = 0
    f = open(answer)
    for line in open(answer):
        token2 = f.readline().split(',')
        content = token2[0]
        idx = speaker.index(content)
        typeidx = anstype.index(str(token2[1].split('\n')[0]))
        aa = [0] * 48
        aa[typeidx] = 1
        speaker[idx] = aa                
        process = process + 1
        if process % 100 ==0:
            print("answer processing to line:  " + str(process))
    f.close()
    return [value_list, speaker] 






		
