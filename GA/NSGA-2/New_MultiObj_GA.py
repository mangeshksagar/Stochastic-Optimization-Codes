#%%
import numpy as np
import pandas as pd
#%%
Ux=5
Lx=-5
num_var=1
size_of_chromo=8
bitsize=size_of_chromo*num_var
popsize=100
crossoverProbability = 0.7
mutationProbability = 1 / popsize
#%%
"Minimize both objective function"
def obj1(x):
    return x**2
#%%
def obj2(x):
    return (x-2)**2
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

    x= (Lx + ((Ux-Lx)/((2**k)-1))*summ)
    return (x)
#%%
def GenFitness(popsize,Lx,Ux):
    x = []
    for i in np.arange(len(popsize)):
        a =   list(popsize[i])
        x1 = Lx + ((Ux - Lx) / (2**bitsize - 1)) * bintod(a)
        x.append(x1)
    return x
#%%
"Upate Pareto Set"
def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 
#%%
def ParetoSetUpdate(ParetoSet , x_new):
    Set = []
    E1 = [obj1(x_new) , obj2(x_new)]
    for i in range(len(ParetoSet)):
        E2 = [obj1(ParetoSet[i]) , obj2(ParetoSet[i])]
        if E1[0] < E2[0] and E1[1] < E2[1]:
           Set.append(x_new)
        elif E1[0] > E2[0] and E1[1] > E2[1]:
            Set.append(ParetoSet[i])
        else:
            Set.append(x_new)
            Set.append(ParetoSet[i])
    return Remove(Set)
#%%
"Rank Examples Function"
def RankSolution(x):
    RankedSolution = []
    while len(x) > 1:
        ParetoSet = [x[0]]
        for i in range(1,len(x)):
            ParetoSet = ParetoSetUpdate(ParetoSet , x[i])
    
        RankedSolution.append(ParetoSet)
        for i in range(len(ParetoSet)):
            x.remove(ParetoSet[i])
    
    if len(x) == 1:
        RankedSolution.append(x)
    
    return RankedSolution
#%%
"Tournament Selection Function"
def TournamentSelection(RankedSolution,popsize):
    FitterSolution = []
    while True:
        a = np.random.randint(len(RankedSolution))
        b = np.random.randint(len(RankedSolution))
        c = RankedSolution[a]
        d = RankedSolution[b]
        p1 = c[np.random.randint(len(c))]
        p2 = d[np.random.randint(len(d))]
        
        if a < b:
            FitterSolution.append(p1)
        else:
            FitterSolution.append(p2)
        
        if len(FitterSolution) == popsize:
            break
        
    return FitterSolution
#%%
"Crossover function"
def Crossover(FitterSolution,bitsize):
    CrossOveredExamples = []
    while True:    
        splitJunction = np.random.randint(bitsize-1)
        p1 = FitterSolution[np.random.randint(len(FitterSolution))]
        p2 = FitterSolution[np.random.randint(len(FitterSolution))] 
        if splitJunction > bitsize:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
            
        if len(CrossOveredExamples) == len(FitterSolution):
            break
    return CrossOveredExamples
#%%
"Mutation function"
def Mutation(CrossOveredExamples,bitsize,mutationProbability):
    mutatePopulation = []
    for j in np.arange(len(CrossOveredExamples)):
        mutationExample = CrossOveredExamples[j]
        flip = []
        for i in np.arange(bitsize):
            if np.random.uniform(0,(mutationProbability)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
    return mutatePopulation
#%%
"Finding fitness"
generation=500
population = []

for i in np.arange(popsize):
    population.append(np.random.randint(low = 0,high = 2,size = bitsize))

for gen in range(generation):
    print(gen)
    "Decode Population"
    x = GenFitness(population , Lx,Ux)
    
    "finding fitness"
    fitness = []
    for i in range(popsize):
        fitness.append([obj1(x[i]) , obj2(x[i])])
    
    "Ranking Solutions"
    RankedSolution = RankSolution(x.copy())
    
    "Tournament Selection"
    FitterSolution = TournamentSelection(RankedSolution,popsize)
    
    newSolution = []
    for i in range(popsize):
        newSolution.append(population[np.where(FitterSolution[i] == x)[0][0]])
    
    "Crossover"
    CrossOveredExamples = Crossover(newSolution,bitsize)
   
    "Mutation"
    mutatePopulation = Mutation(CrossOveredExamples,bitsize,mutationProbability)
    
    population = mutatePopulation
#%%
x = GenFitness(population ,Lx,Ux)
fitness = []
for i in range(popsize):
    fitness.append([obj1(x[i]) , obj2(x[i])])    
#%%