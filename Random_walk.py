#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:59:57 2019

@author: mangesh.kshirsagar
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 4 * x - 4
#%%
x_old = np.random.random()
E1 = f(x_old)
x = []
energy = []
for i in np.arange(1000):
    h = 0.2 * np.random.uniform() # h is the random Step
    x_new = x_old + h 
    E2 = f(x_new)
    x_old = x_new    
    print("x is %f and energy is %f"%(x_old,E2))
    x.append(x_new)
    energy.append(E2)
    plt.plot(x,energy)
plt.show()
ind = np.argmin(energy)
x_min = x[ind]
print("The Minimum Value is:",x_min)
#%%