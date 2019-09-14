import numpy as np
import pandas as pd
import random
import scipy.stats as ss

def f(x1,x2,x3):
	y=(x1**2)*x2*(x3+2)
	return y

def constrain1(x1,x2,x3):
	y=1- (((x2**3)*x3)/(71785*(x1**4)))
	return y

def constrain2(x1,x2,x3):
	y=((4*x2**2 - x1*x2)/(12566*((x1**3)*x2 - (x1**4)))) + (1/(5108*(x1**2))) -1
	return y

def constrain3(x1,x2,x3):
	y=1-(140.45*x1)/(x3*x2**2)
	return y

def constrain4(x1,x2,x3):
	y=(x1+x2)/1.5 - 1
	return y

###########
def chooseX(x1,x2,x3):
	c1=constrain1(x1,x2,x3)
	c2=constrain2(x1,x2,x3)
	c3=constrain3(x1,x2,x3)
	c4=constrain4(x1,x2,x3)
	C_arr=np.array([c1,c2,c3,c4])
	#print(C_arr)
	#feasible=pd.DataFrame(columns=['X1','X2','X3','F'])
	#Non_feasible=pd.DataFrame(columns=['X1','X2','X3','F', 'Num voilation'])
	constrain_voilation=sum([max(0,C_arr[i]) for i in range(len(C_arr))])

	#print(constrain_voilation)

	if c1<=0 and c2<=0 and c3<=0 and c4<=0:
		num_voilation=0

		print("Hora")
		return pd.DataFrame([[x1,x2,x3,f(x1,x2,x3),num_voilation,constrain_voilation]],columns=['X1','X2','X3','F','Num voilation','Constrain Voilation'])
		#return(x1,x2,x3,f(x1,x2,x3))
	else:
		num_voilation=len(np.where(C_arr>0)[0]) ####/len(C_arr)
		return pd.DataFrame([[x1,x2,x3,f(x1,x2,x3),num_voilation,constrain_voilation]],columns=['X1','X2','X3','F','Num voilation','Constrain Voilation'])
###########


############
def create_init_pops():
	X=pd.DataFrame(columns=['X1','X2','X3','F', 'Num voilation','Constrain Voilation'])
	while(True):
		x1=np.random.uniform(0,10)
		x2=np.random.uniform(0,10)
		x3=np.random.uniform(0,20)
		#F=f(x1,x2,x3)
		eval=chooseX(x1,x2,x3)
		if eval.empty==False:
			X=X.append(eval,ignore_index=True)

		if len(X)==PopSize:
			break

	#df=pd.DataFrame({'X1':X1,'X2':X2,'X3':X3,'F':F})
	return X
##############


def create_next_pops(XX):
	X=pd.DataFrame(columns=['X1','X2','X3','F', 'Num voilation','Constrain Voilation'])
	i=0
	while(True):
		#x1=np.random.uniform(0,10)
		#x2=np.random.uniform(0,10)
		#x3=np.random.uniform(0,20)
		#F=f(x1,x2,x3)
		eval=chooseX(XX.iloc[i,0],XX.iloc[i,1],XX.iloc[i,2])
		if eval.empty==False:
			X=X.append(eval,ignore_index=True)

		if len(X)==PopSize:
			break
		i=i+1

	#df=pd.DataFrame({'X1':X1,'X2':X2,'X3':X3,'F':F})
	return X

##############
def select_pops(X):
	newpop=pd.DataFrame(columns=['X1','X2','X3'])
	while(len(newpop)<PopSize):
		choice=X.sample(2)
		n=np.where(choice['Num voilation']==0)[0]
		violations_in_choice=choice['Num voilation']

		if len(n)>=1:
			newpop=newpop.append(choice)
		elif(violations_in_choice.iloc[0]!=violations_in_choice.iloc[1]):
			newpop=newpop.append(choice.iloc[np.where(choice['Num voilation']==choice['Num voilation'].min())[0]].iloc[:,0:3])
		elif(violations_in_choice.iloc[0]==violations_in_choice.iloc[1]):
			newpop=newpop.append(choice.iloc[np.where(choice['Constrain Voilation']==choice['Constrain Voilation'].min())[0]].iloc[:,0:3])
	return newpop.reset_index(drop=True)
##############



###############
def crossover(X):
	crossovered=pd.DataFrame()

	for i in range(PopSize):
		c=pd.DataFrame()
		choic=np.random.choice(range(len(X)),2)
		parent1=X.iloc[choic[0]]
		#print(parent1)
		#print(parent1[0])

		p1=chooseX(parent1[0],parent1[1],parent1[2])

		parent2=X.iloc[choic[1]]
		p2=chooseX(parent2[0],parent2[1],parent2[2])

		child1=0.5*parent1 + 0.5*parent2
		if (child1[0]>0 and child1[0]<10) and (child1[1]>0 and child1[1]<10) and  (child1[2]>0 and child1[2]<20):
			ch1=chooseX(child1[0],child1[1],child1[2])
		else:

		child2=1.5*parent1 - 0.5*parent2
		ch2=chooseX(child2[0],child2[1],child2[2])

		child3=0.5*parent1 + 1.5*parent2
		ch3=chooseX(child3[0],child3[1],child3[2])

		#crossovered=crossovered.append([parent1,parent2,child1,child2,child3],ignore_index=True)
		c=c.append([p1,p2,ch1,ch2,ch3],ignore_index=True)
		#print(i)

		print(c)

		nn=len(c.iloc[np.where(c['Num voilation']==c['Num voilation'].min())[0]])

		if nn>=2:
			crossovered=crossovered.append(c.iloc[np.where(c['Num voilation']==c['Num voilation'].min())[0]].sort_values(by='F').iloc[0:2,:],ignore_index=True)
		elif nn==1:
			crossovered=crossovered.append(c.iloc[np.where(c['Num voilation']==c['Num voilation'].min())[0]],ignore_index=True)
			crossovered=crossovered.append(c.iloc[np.where(c['Constrain Voilation']==c['Constrain Voilation'].min())[0]],ignore_index=True)
		crossovered=crossovered.drop_duplicates()
		#if len(crossovered)==




	return (crossovered)
###############

##############
def mutation(crossover_pops,mu):
	for i in range(PopSize):
		sd_X=crossover_pops.describe().loc['std'][0:3]
		mean_X=crossover_pops.describe().loc['mean'][0:3]
		if np.random.uniform()<mu:
			crossover_pops['X1'].iloc[i]=crossover_pops['X1'].iloc[i]+ ss.norm.pdf(0,sd_X[0])
		if np.random.uniform()<mu:
			crossover_pops['X2'].iloc[i]=crossover_pops['X2'].iloc[i]+ ss.norm.pdf(0,sd_X[1])
		if np.random.uniform()<mu:
			crossover_pops['X3'].iloc[i]=crossover_pops['X3'].iloc[i]+ ss.norm.pdf(0,sd_X[2])
	return crossover_pops
#############

PopSize=100

x1=np.random.uniform(0,10,PopSize)
x2=np.random.uniform(0,10,PopSize)
x3=np.random.uniform(0,20,PopSize)

X=pd.DataFrame({'X1':x1,'X2':x2,'X3':x3})

ff=np.vectorize(f)
c1=np.vectorize(constrain1)

f_vector=ff(x1,x2,x3)
c_vector=c1(x1,x2,x3)

X1=pd.Series()
X2=pd.Series()
X3=pd.Series()
F=pd.Series()
i=0

#generate initial population

#df=X.sort_values(by=['F']).reset_index(drop=True)


init_pop= create_init_pops().sort_values(by='F').reset_index(drop=True)

#while(True):
while (True):
	voilation=init_pop['Num voilation']
	funcs=init_pop['F']
	X=init_pop.drop(['F', 'Num voilation' ],axis=1)

	#print(X)

	selection_pool=select_pops(init_pop)
	crossover_pops=crossover(selection_pool).sort_values('Num voilation').iloc[0:PopSize].reset_index(drop=True)
	#print(crossover_pops)
	mu=0.1
	mutate_pops=mutation(crossover_pops,mu)
	#print(mutate_pops)
	init_pop=create_next_pops(mutate_pops)
	#print(init_pop)
	#print("hello")
