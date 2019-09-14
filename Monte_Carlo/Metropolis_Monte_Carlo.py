
"""
Created on Sun Aug 11 00:31:25 2019

@author: mangesh.kshirsagar
"""
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
#%%
Up_limx=5
Lo_limx=-5
Up_limy=5
Lo_limy=-5
colors = "bgrcmykw"
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
X=np.random.uniform(Lo_limx,Up_limx,1000)
Y=np.random.uniform(Lo_limy,Up_limy,1000)
F=F_X(X,Y)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, F, cmap=plt.cm.viridis, linewidth=0.2)
plt.title("Graphical Representation of Rosenbrock")

ax.scatter(-2 ,2,1)
plt.ion()
#%%


h= 0.08

#plt.show()

#def checklims(xold,yold):
    #if xold>Up_limx or xold<Lo_limx or yold>Up_limy or yold<Lo_limY:


count=1
while(count<50):
    xold=np.random.uniform(Lo_limx,Up_limx,1)
    yold=np.random.uniform(Lo_limy,Up_limy,1)
    fold=F_X(xold,yold)
    ax.plot([xold,xold], [yold,yold],[0,fold],linewidth=2)
    for i in range(5):
        for j in range(5):
            for k in range(5):

                while(True):
                    P=np.random.sample()
                    if P<0.5:
                        xnew=xold+h
                        ynew=yold+h
                        if xnew<Up_limx and xnew>Lo_limx and ynew<Up_limy and ynew>Lo_limy:
                            fnew=F_X(xnew,ynew)
                            if fnew-fold<0: # for minimization
                                xold=xnew
                                yold=ynew
                                fold=fnew
                            break
                    else:
                        xnew=xold-h
                        ynew=yold-h
                        if xnew<Up_limx and xnew>Lo_limx and ynew<Up_limy and ynew>Lo_limy:
                            fnew=F_X(xnew,ynew)
                            if fnew-fold<0: # for minimization
                                xold=xnew
                                yold=ynew
                                fold=fnew

                            break

                #checklims(xold,yold)


                fold=F_X(xold,yold)
                #print(xold,yold,fold)

                ax.plot([xold,xold], [yold,yold],[0,fold],c=colors[int(np.random.randint(0,8,1))],linewidth=2)
                #sc._offset3d=(xnew,ynew,fnew)
                plt.show()
                plt.pause(0.1)
                del ax.lines[1]

                #plt.clf()


    print(xold,yold,fold)
    #print(count)
    count=count+1
#%%
#ani=matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=40, blit=False)