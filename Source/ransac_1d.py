import random
import matplotlib.pyplot as plt
import numpy as np
import math
# -----------------------------------------------------------------

    #-------------------------Datafile Path------------------------
inFile='input.txt'
outFile='output.txt'
path = '/home/korus/hg_funcs/Data/RANSAC/linear/'
    #-------------------------RANSAC Parameter------------------------
threshold = 1
N = 25
m = 2
    #-------------------------Model Parameter------------------------
    # model = ax+by+c (linear function)
a = 0
b = -1
c = 0

# -----------------------------------------------------------------

def fileRead(x,y):
    print('--------------File Read--------------------------')
    f=open(path+inFile)
    while True:
        line = f.readline()
        if not line: break
        words = line.split(' ')
        x.append(float(words[0]))
        y.append(float(words[1]))
    f.close()

def fileWrite(x,y):
    print('--------------File Write--------------------------')
    f=open(path+outFile,'w')

    f.write(str(x[0])+' '+str(y[0]))
    for i in range(1, len(x)):
        f.write('\n'+str(x[i])+' '+str(y[i]))

def getMin(list):
    min =list[0]
    for i in range(len(list)):
        if min > list[i]:
            min = list[i]
    return min

def getMax(list):
    max =list[0]
    for i in range(len(list)):
        if max < list[i]:
            max = list[i]
    return max

def getRandomSample(set, data_size):
    print('--------------getSample Read--------------------------')
    while len(set) != m:
            ran = random.randint(0,data_size-1)
            set.add(ran)

def getModel(x,y,sample):
    global a
    global c
    # print('--------------getModel--------------------------')
    # print(sample[0])
    # print(y[sample[0]])
    # print(sample[1])
    # print(y[sample[1]])
    # print('sample size = {}'.format(len(sample)))
    # print('data size = {}'.format(len(x)))
   
    a = (float)(y[sample[1]]-y[sample[0]]) / (float)(x[sample[1]]-x[sample[0]])
    c = -a*x[sample[0]] +y[sample[0]]
    print(a)
    print(c)
def getInlier(model_a, model_b, model_c, x, y):
    num=0
    for i in range(len(x)):
        dist = (model_a * x[i] + model_b * y[i] + model_c) / math.sqrt(model_a*model_a + model_b * model_b)
        if dist < threshold:
            num = num+1
    return num
def RANSAC(x,y):
    max=0
    for i in range(N):
        sample = set([])
        getRandomSample(sample,len(x))
        getModel(x, y, list(sample))
        num = getInlier(a, b, c, x, y)
        if num > max:
            max=num
            re_a=a
            re_b=b
            re_c=c
    return re_a, re_b, re_c
    

def printGraph(x, y, model_x, model_y):
    plt.plot(x,y,'*b')
    plt.plot(model_x, model_y,'*r')
    plt.xticks(np.arange(getMin(x)-10, getMax(x)+10, 10))
    plt.yticks(np.arange(getMin(y)-10, getMax(y)+10, 10))
    plt.show()
    
def main():
    model_a = 0
    model_b = -1
    model_c = 0
    x=[]
    y=[]
    fileRead(x,y)
    if len(x) == 0:
        print('No File Data')
        return
    model_a, model_b, model_c = RANSAC(x,y)

    print('----------MODEL-----------')
    print(model_a)
    print(model_b)
    print(model_c)

    model_x=np.arange(x[len(x)-1],x[0],0.1)
    print(x[0])
    print(x[len(x)-1])
    print('SIZZE = {}'.format(len(x)))
    print('SIZZE = {}'.format(len(model_x)))

    model_y = model_a/model_b*model_x*-1 + model_c/model_b *-1
    fileWrite(model_x,model_y)

    printGraph(x, y, model_x, model_y)
    
if __name__ == "__main__":
    main()