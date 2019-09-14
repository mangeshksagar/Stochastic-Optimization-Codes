#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import random
import math
#%%
T=500

Ux=4
Lx=-1.5
Uy=4
Ly=-3
num_var=2
#%%
def F_X(x,y):
    #F1=4*x*x +4*y*y
    #F1=math.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1
    #F1=x*xy-x
    F1=100*((y-x**2)**2) + (1-x)**2
    #F1=x+y
    #F2=(x-5)*(x-5) + (y-5)*(y-5)
    #F2=(x-2)*(x-2)
    return(F1)
#%%
x_old=np.random.uniform(Lx,Ux,1)
y_old=np.random.uniform(Ly,Uy,1)
f_old=F_X(x_old,y_old)
while (T>0.00001):
    for i in range(50):
        x_new=np.random.uniform(Lx,Ux,1)
        y_new=np.random.uniform(Ly,Uy,1)

        f_new=F_X(x_new,y_new)


        Del_E= f_new-f_old
        P=math.exp(-Del_E/T)
        r=np.random.sample()
        if (Del_E<0):
            x_old=x_new
            y_old=y_new

        else:
            if r<P:
                x_old=x_new
                y_old=y_new

        print(x_old,y_old,f_new)
        #print(P)
    print(T)
    T=T-0.002
#%%