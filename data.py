#Imagawa Yuuki
import sys
import os
import numpy as np

def read_data(d):
    f=open('output-0.3.csv','r')
    line=f.readlines()
    f.close()
    tmp=np.loadtxt(line)
    N=tmp.shape[0]
    x=np.zeros(N)
    y=np.zeros(N)
    for i in range(N):
        x[i]=tmp[i,0]
        y[i]=tmp[i,1]
    return x,y
