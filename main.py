#Imagawa Yuuki
import sys
import os
import numpy as np
import argparse
import random
import argparse
from data import read_data
from matplotlib import pyplot as pyp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from programs import *
from wrt import * 

#estimate_hyper_perameter
parser=argparse.ArgumentParser()
parser.add_argument('--d',type=int,default=1,help='dimention of input x (default=2)')
parser.add_argument('--C',type=float,default=1,help='regurality parameter (default=1)')
parser.add_argument('--Nitr',type=int,default=1000,help='Total number of iteration step (default=1000)')
parser.add_argument('--ε',type=float,default=0.1,help='regurality parameter (default=0.1)')
parser.add_argument('--γ',type=float,default=1,help='regurality parameter (default=1)')

args = parser.parse_args()
d=args.d
C=args.C
Nitr=args.Nitr
ε=args.ε
γ=args.γ
N=500
N_train=int(N*0.7)
N_test=N-N_train


lambda_array=np.zeros(N_train)
omega_array=np.zeros(Nitr)
p_train=np.zeros((N_train,d))
p_test=np.zeros((N_test,d))
fx=np.zeros(N_train)
b=np.zeros(N_train)
X_train=np.zeros(N_train)
y_train=np.zeros(N_train)

#read_date
(x,y)=read_data(d)
x=x.reshape(N,1)
y=y.reshape(N,1)

#split_data
x_train,x_test,y_train,y_test=split_data(x,y,N,N_train)

#dimensinaol_expansion_data
(p_train,p_test)=dimensional_expansion(d,p_train,p_test,x_train,x_test,N_train,N_test)

#make_lernel_matrix
Kernel_matrix=make_Kernel_matrix(p_train,γ,N_train,d) 

#make_fx
fx=make_fx(fx,lambda_array,Kernel_matrix,b,N_train)
u=random.randint(0,N_train-1)
v=random.randint(0,N_train-1)
while u==v:
        v=random.randint(0,N_train-1)


i=0
while i<Nitr:
        #u=random.randint(0,N_train-1)
        #v=random.randint(0,N_train-1)
        #while u==v:
           #v=random.randint(0,N_train-1)
	
	#update_lambda
        lambda_array=update_lambda_array(u,v,lambda_array,Kernel_matrix,ε,y_train,fx,N_train,i,C)
        
        #update_omega
        (omega)=update_omega(lambda_array,y_train,Kernel_matrix,N_train,ε)
        #make_omegaarray
        omega_array[i]=omega
        print(i,omega_array[i])
        #update_fx
        (fx,tmp)=update_fx(fx,lambda_array,Kernel_matrix,b,N_train)
        (I_up,I_low)=update_I_up_I_low(lambda_array,C,N_train,y_train)
        ##update_u
        (U_up,U_low)=update_U_up_U_low(I_up,I_low,tmp)
        ##update_U_up_U_low
        (U_up,U_low)=update_U_up_U_low(I_up,I_low,tmp)
        ##update_y_up_y_low
        (y_up,y_low)=update_y_up_y_low(I_up,I_low,y_train)
        ##update_lambda_up_lambda_low
        (lambda_up,lambda_low)=update_lambda_up_lambda_low(I_up,I_low,lambda_array)
        (F,G)=update_F_G(I_up,I_low,y_up,y_low,U_up,U_low,ε,lambda_up,lambda_low,N_train)
        (u,v)=select_u_v(I_up,I_low,N_train,F,G)
        i=i+1
#make_omega_graph
#make_omega_graph(Nitr,omega_array)

#estimate_b
b=estimate_b(C,N_train,b,ε,y_train,lambda_array,Kernel_matrix)

#estimate_fx
fx=estimate_fx(d,p_train,p_test,N_train,N_test,γ,lambda_array,y_train,b)

#evaluate_result
(MAE,RMSE)=evalution_result(y_test,fx)
print('# RMSE',RMSE)

#plot_fx
plot_fx(fx,N_test,x_test,y_test)
wrt_fx(fx,x_test,y_test,ε)

