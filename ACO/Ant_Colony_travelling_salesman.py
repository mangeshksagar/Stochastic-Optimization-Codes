import numpy as np
import pandas as pd
import copy
max_iter = 50 ; q0  = 0.7
alpha = 1.1; beta = 1.2
cities=[1,2,3,4]
#%%
a = 1/(np.random.uniform(10,100,size=(4,4)))
matrix = np.tril(a) + np.tril(a, -1).T
np.fill_diagonal(matrix,0)

pheromone = np.random.uniform(0.1,1,size=(4,4))
pheno_close_prod = np.multiply(pheromone,matrix)

sum_of_mat = sum(sum(pheno_close_prod ))
normalize_mat  = 1/sum_of_mat * (np.ones(matrix.shape))

probabilistic_mat = np.multiply(normalize_mat, pheno_close_prod)

y= 1/(su(pheno_close_prod))
#%%
A = np.random.randint(0,3)
cityA=[]
cityA.append(A)        
for i in np.arange(3):
    r = np.random.uniform(0,1)        
    if r < q0:
        c =  np.argmax(pheno_close_prod[:,A])
        cityA.append(c)
        pheno_close_prod[:,A]=0
        pheno_close_prod[A]=0
        probabilistic_mat[:,A]=0
        probabilistic_mat[A]=0   
        A=c
        
    else:
        c = np.argmax(probabilistic_mat[:,A])
        cityA.append(c)
        pheno_close_prod[:,A]=0
        pheno_close_prod[A]=0
        probabilistic_mat[:,A]=0
        probabilistic_mat[A]=0   
        A=c
