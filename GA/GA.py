#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import random
import math
#%%
Ux=4
Lx=-1.5
Uy=4
Ly=-3
num_var=2
size_of_chromo=8
bitsize=size_of_chromo*num_var
popsize=100
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
##generate  initial binary population
def create_init_pop(popsize):
    arr=pd.Series(np.arange(0,popsize))
    j=lambda i:np.random.binomial(1,0.5,bitsize)
    arr=arr.apply(j)
    return arr

#%%
def bintod(binary_number):

    k=8
    summ=sum((2**i)*binary_number[i] for i in range(0,0+k))

    x1= (Lx + ((Ux-Lx)/((2**k)-1))*summ)

    summ=0
    summ=sum((2**(i-k))*binary_number[i] for i in range(k,2*k))

    y1= (Ly + ((Uy-Ly)/((2**k)-1))*summ)
    return (x1,y1)
#%%
def tournament_selection(fun_values,Binary_pops):
    selected_pool=pd.Series()
    #selected_pool=[]
    #selected_pool=
    for i in range(len(fun_values)):
        choice=random.sample(range(len(fun_values)),2)
        ran_choice=np.where(fun_values==min(fun_values[choice])) [0]
        #selected_pool=selected_pool.set_value(i,Binary_pops[ran_choice])
        #selected_pool.append(Binary_pops[ran_choice])
        selected_pool=pd.concat([selected_pool,Binary_pops[ran_choice]], axis=0,ignore_index=True)
    return selected_pool
#%%

def crossover(pool):
    updatedpool=pd.Series()
    child1=pd.Series()
    child2=pd.Series()
    total_ch=pd.Series()

    for i in range(0,popsize):
        choice=random.sample(range(popsize),2)
        rand_slct_pool=pool[choice]
        #print(rand_slct_pool)
        fixnumber= np.random.randint(0,bitsize)
        tempx1=rand_slct_pool.iloc[0][fixnumber:]
        #print(tempx1)
        tempx2=rand_slct_pool.iloc[1][fixnumber:]
        child1x=np.append(rand_slct_pool.iloc[1][:fixnumber],tempx1)
        child2x=np.append(rand_slct_pool.iloc[0][:fixnumber],tempx2)
        #print(child2x)

        #child1=child1.append(child1x)
        #child=child.append(child2x)
        child1=child1.set_value(i,child1x)
        child2=child2.set_value(i,child2x)
        #total_ch=total_ch.append(child1)
        #total_ch=total_ch.append(child2, ignore_index=True)
    updatedpool=pd.concat([child1,child2], axis=0,ignore_index=True)
    #updatedpool=pd.concat([updatedpool,child1], axis=0,ignore_index=True)
    return updatedpool
#%%

def mutation(crossoverpops,mu_rate):
    for i in range(len(crossoverpops)):
        for k in range(len(crossoverpops[i])):
            if np.random.sample()<mu_rate:
                crossoverpops[i][k]=abs(crossoverpops[i][k]-1)
    return crossoverpops
#%%


Pops=create_init_pop(popsize) ##generate  initial binary population
B2D=np.vectorize(bintod)
fitness=np.vectorize(F_X)

for i in range(1000):
    df=pd.DataFrame(columns=['X','Y','Fx'])
    Bin2D=B2D(Pops)
    df['X']=Bin2D[0]
    df['Y']=Bin2D[1]


    #Evaluate fitness function

    fun_values=pd.Series(fitness(Bin2D[0],Bin2D[1]))
    df['Fx']=fun_values
    print(df.head(5))

    #tournament selection
    selection_pool=tournament_selection(fun_values,Pops)

    crossover_pops=crossover(selection_pool)
    mu_rate=0.1
    mutate_pops=mutation(crossover_pops,mu_rate)
    total_pop=pd.concat([Pops,mutate_pops], axis=0,ignore_index=True)
    Bin2D1=B2D(total_pop)
    fun_values_ind=(pd.Series(fitness(Bin2D1[0],Bin2D1[1])).sort_values().index)[0:popsize]

    Pops=total_pop[fun_values_ind].reset_index(drop=True)

#fitness=fitness_func(pops)
#%%