import numpy as np
import pandas as pd
import random
import math

def FX(x):
    if (x<=1):
        F1=-x
    elif(x>1 and x<=3):
        F1=x-2
    elif(x>3 and x<=4):
        F1=4-x
    elif(x>4):
        F1=x-4
    F2=(x-5)**2
    return (F1,F2)

#############

lo_lim=-5
up_lim=10
Pareto_front=[]

x_init=np.random.uniform(lo_lim,up_lim,1)
T=1
while(T<10):
    funcs_old=FX(x_init)
    Pareto_front.append((x_init,funcs_old))
    x_new=np.random.uniform(lo_lim,up_lim,1)
    funcs_new=FX(x_new)
    T=T+1
