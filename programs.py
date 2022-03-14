#Imagawa Yuuki
import numpy as np
import sys
import random
from matplotlib import pyplot as pyp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def split_data(x,y,N,N_train):
   xy=np.concatenate([x,y],1)
   np.random.shuffle(xy)

   xy_train, xy_test=train_test_split(xy, test_size=0.3)
   x_train,y_train=np.hsplit(xy_train,2)
   x_test,y_test=np.hsplit(xy_test,2)
   return x_train,x_test,y_train,y_test


def dimensional_expansion(d,p_train,p_test,x_train,x_test,N_train,N_test):
   i=0
   j=0
   for i in range(N_train):
     for j in range(d):
      p_train[i,j]=np.power(x_train[i],j+1)
   for i in range(N_test):
     for j in range(d):
       p_test[i,j]=np.power(x_test[i],j+1)
   return p_train, p_test
   

def make_Kernel_matrix(p_train,γ,N_train,d):
   p_train=p_train.reshape(d,N_train)
   p2=np.power(p_train,2)
   Z=np.sum(p2,axis=0)
   Z=Z.reshape(N_train,1)
   A=np.tile(Z,(1,N_train))
   Z=Z.reshape(1,N_train)
   B=np.tile(Z,(N_train,1))
   D=np.dot(p_train.T,p_train)
   K=A+B-2*D
   Kernel_matrix=np.exp(-γ*K)
   return Kernel_matrix


def make_fx(fx,lambda_array,Kernel_matrix,b,N_train):
   i=0
   for i in range(N_train):
     tmp1=np.dot(lambda_array.T,Kernel_matrix)
     fx=tmp1+b
   return fx    
 
 
def update_lambda_array(u,v,lambda_array,Kernel_matrix,ε,y_train,fx,N_train,i,C):
   s=lambda_array[u]+lambda_array[v]
   η=Kernel_matrix[u,u]-2*Kernel_matrix[u,v]+Kernel_matrix[v,v]
   delta=2*ε/η
   tmp1=y_train[v]-y_train[u]+fx[u]-fx[v]
   lambda_array[v]=lambda_array[v]+tmp1/η
   lambda_array[u]=s-lambda_array[v]
   if lambda_array[u]*lambda_array[v]<0:
      if abs(lambda_array[u])>=delta and abs(lambda_array[v])>=delta:
          lambda_array[v]=lambda_array[v]-np.sign(lambda_array[v])*delta
      else:
          if abs(lambda_array[v])>abs(lambda_array[u]):
             lambda_array[v]=s
          if abs(lambda_array[v])<abs(lambda_array[u]):
             lambda_array[v]=0
   L=max(s-C,-C)
   H=min(C,s+C)
   if lambda_array[v]<L:
      lambda_array[v]=L
   if lambda_array[v]>H:
      lambda_array[v]=H
   lambda_array[u]=s-lambda_array[v]
   return lambda_array
   
def update_omega(lambda_array,y_train,Kernel_matrix,N_train,ε):
   lambda_array=lambda_array.reshape(1,N_train)
   y_train=y_train.reshape(1,N_train)

   tmp2=np.dot(lambda_array,y_train.T)
   tmp3=np.abs(lambda_array)
   tmp4=ε*np.sum(tmp3)
   tmp5=np.dot(lambda_array,Kernel_matrix)
   tmp6=np.dot(tmp5,lambda_array.T)
   omega=0.5*tmp6-tmp2+tmp4
   return omega

 
def update_fx(fx,lambda_array,Kernel_matrix,b,N_train):
   l=0
   for l in range(N_train):
     tmp=np.dot(lambda_array.T,Kernel_matrix)
     fx[l]=tmp[l] 
   return fx,tmp

def update_I_up_I_low(lambda_array,C,N_train,y_train):
    I_up=[]
    I_low=[]
    i=0
    for i in range(N_train):
        if (lambda_array[i]>=-C and lambda_array[i]<C):
            I_up.append(i)
        if (lambda_array[i]>-C and lambda_array[i]<=C):
            I_low.append(i)
    return I_up,I_low

def update_U_up_U_low(I_up,I_low,tmp):
    U_up=np.zeros(len(I_up))
    U_low=np.zeros(len(I_low))
    I_up=np.array(I_up)
    I_low=np.array(I_low)
    i=0
    for i in range(len(I_up)):
        U_up[i]=tmp[I_up[i]]
    i=0
    for i in range(len(I_low)):
        U_low[i]=tmp[I_low[i]]
    return U_up,U_low

def update_y_up_y_low(I_up,I_low,y_train):
    y_up=np.zeros(len(I_up))
    y_low=np.zeros(len(I_low))
    I_up=np.array(I_up)
    I_low=np.array(I_low)
    i=0
    for i in range(len(I_up)):
        y_up[i]=y_train[I_up[i]]
    i=0
    for i in range(len(I_low)):
        y_low[i]=y_train[I_low[i]]
    return y_up,y_low

def update_lambda_up_lambda_low(I_up,I_low,lambda_array):
    lambda_up=np.zeros(len(I_up))
    lambda_low=np.zeros(len(I_low))
    I_up=np.array(I_up)
    I_low=np.array(I_low)

    i=0
    for i in range(len(I_up)):
        lambda_up[i]=lambda_array[I_up[i]]
    i=0
    for i in range(len(I_low)):
        lambda_low[i]=lambda_array[I_low[i]]
    return  lambda_up,lambda_low

def update_F_G(I_up,I_low,y_up,y_low,U_up,U_low,ε,lambda_up,lambda_low,N_train):
    y_up=y_up.reshape(len(I_up),1)
    y_low=y_low.reshape(len(I_low),1)
    U_up=U_up.reshape(len(I_up),1)
    U_low=U_low.reshape(len(I_low),1)
    lambda_up=lambda_up.reshape(len(I_up),1)
    lambda_low=lambda_low.reshape(len(lambda_low),1)
    F=np.zeros(len(I_up))
    G=np.zeros(len(I_low))
    i=0
    for i in range(len(I_up)):
        F[i]=y_up[i,0]-U_up[i]-ε*np.sign(lambda_up[i])
    i=0
    for i in range(len(I_low)):
        G[i]=y_low[i,0]-U_low[i]-ε*np.sign(lambda_low[i])
    return F,G

def select_u_v(I_up,I_low,N_train,F,G):
    u=np.argmax(F)
    u=I_up[u]
    v=np.argmin(G)
    v=I_low[v]
    while u==v:
      t=random.randint(0,N_train-1)
    return u,v

def make_omega_graph(Nitr,omega_array):
   Iitr=np.zeros(Nitr)
   i=0
   for i in range(Nitr):
       Iitr[i]=i

   pyp.title("SVR_omega-graph",{"fontsize":25})
   pyp.xlabel("Niitr",{"fontsize":15})
   pyp.ylabel("omega",{"fontsize":15})
   pyp.plot(Iitr,omega_array)
   pyp.show()
   return Iitr

def estimate_b(C,N_train,b,ε,y_train,lambda_array,Kernel_matrix):
   lambda_array=lambda_array.reshape(1,N_train)
   Kernel_matrix=Kernel_matrix.reshape(N_train,N_train)
   I_for_b=[]
   i=0
   for i in range(N_train):
       if lambda_array[0,i]>-C and lambda_array[0,i]<0:
           b[i]=ε+y_train[i]-np.dot(lambda_array,Kernel_matrix[i])
           I_for_b.append(i)
       if lambda_array[0,i]>0 and lambda_array[0,i]<C:
           b[i]=-ε+y_train[i]-np.dot(lambda_array,Kernel_matrix[i])
           I_for_b.append(i)
   b_for_b=np.zeros(len(I_for_b))
   i=0
   for i in range(len(I_for_b)):
       b_for_b[i]=b[I_for_b[i]]
   b=sum(b_for_b)/len(b_for_b)
   return b
 

def estimate_fx(d,p_train,p_test,N_train,N_test,γ,lambda_array,y_train,b):
    p_test=p_test.reshape(d,N_test)
    p2=np.power(p_test,2)
    Z=np.sum(p2,axis=0)
    Z=Z.reshape(1,N_test)
    A=np.tile(Z,(N_train,1))
   
    p_train=p_train.reshape(d,N_train)
    p2=np.power(p_train,2)
    Z=np.sum(p2,axis=0)
    Z=Z.reshape(1,N_train)
    B=np.tile(Z,(N_test,1))
    B=B.T
    D=np.dot(p_train.T,p_test)
    E=A+B-2.0*D
    Kernel=np.exp(-γ*E) 
    
    lambda_array=lambda_array.reshape(1,N_train)
    F=np.dot(lambda_array,Kernel)
    fx=F+b
    return fx

def plot_fx(fx,N_test,x_test,y_test):
     
    X=np.zeros(N_test)
    y=np.zeros(N_test)
    Y=np.zeros(N_test)
    i=0
    for i in range(N_test):
        X[i]=x_test[i]
        y[i]=y_test[i]
        Y[i]=fx[0,i]


def evalution_result(y_test,fx):
    y_test=y_test.T
    MAE=mean_absolute_error(y_test,fx)
    RMSE=np.sqrt(mean_squared_error(y_test,fx))
    return MAE,RMSE

