import sys,os 
import numpy as np

def wrt_fx(fx,x_test,y_test,ε):
    N=x_test.shape[0]  
    fx=fx.flatten() 
    x_fx_y=np.zeros((N,3)) 
    for i in range(N):
        x_fx_y[i,0]=x_test[i]
        x_fx_y[i,1]=fx[i]
        x_fx_y[i,2]=y_test[i] 
    f=open('x_fx_y.txt','w')
    for i in range(N): 
        for k in range(x_fx_y.shape[1]): 
            f.write(str(x_fx_y[i,k]))   
            f.write('\t')
        f.write('\n') 

    N=x_test.shape[0]
    fx=fx.flatten()
    x_fx_y_eps=np.zeros((N,3))
    for i in range(N):
        x_fx_y_eps[i,0]=x_test[i]
        x_fx_y_eps[i,1]=fx[i]+ε
        x_fx_y_eps[i,2]=fx[i]-ε
    f=open('x_fx_y_eps.txt','w')
    for i in range(N):
        for k in range(x_fx_y.shape[1]):
            f.write(str(x_fx_y_eps[i,k]))
            f.write('\t')
        f.write('\n')

