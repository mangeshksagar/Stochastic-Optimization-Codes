#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,make_scorer
import copy
import warnings
warnings.filterwarnings("ignore")
#%%
data = pd.read_csv("/nfs/cms/mtech18/mangesh.kshirsagar/Courses/Mtech/sem-2/ML CODE/DATASETS/wdbc.csv")

y=data["diagnosis"]
y.replace('M',0,inplace=True)
y.replace('B',1,inplace=True)

data = data.drop(data.columns[[0,1,32]], axis=1)
data.isnull().sum()

#%%
popln = 10
Pc = 0.6
Pm = 1 / popln
bits = len(data.columns)

p=[]
for i in range(popln):
   a = np.random.choice([0, 1], size=bits)
   p.append(a)

ACC_vec=[];BEST_ACC_vec=[]; attri=[]
for U in range(500): 
    
    ACC =[] ; FIT =[]
    indx = [] ; D=[]
    for i in np.arange(len(p)): 
        ind = list(np.where(p[i]==1))
        indx.append(ind)
        I = indx[0][0]
        
        test = data[data.columns[list(I)]]
        X_train,X_test,y_train,y_test = train_test_split(test,y,test_size = 0.2,random_state = 42)
        model = RandomForestClassifier()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        m_acc = accuracy_score(y_test,y_pred)

        ACC.append(m_acc) 
        c = np.linspace(0.2,1,5)
        fit=[]
        for j in np.arange(len(c)):     
            fitnes = m_acc - (j*len(I) / len(data.columns))
            fit.append(fitnes)
            F = np.max(fit)
        FIT.append(F)
 #%%       
    #Tournament selection    
    Best_fit=[]
    random= np.random.randint(0, 10, size=(popln,2))
    for k in np.arange(popln):
        maximum = min(FIT[random[k][0]],FIT[random[k][1]])
        index = FIT.index(maximum)
        Best_fit.append(index)
    
    #finding Best chromosomes from original popln with help of tournament selection
    Best_population=[]
    for m in np.arange(popln):
        B = p[Best_fit[m]]
        Best_population.append(B)
    
    #    
    random= np.random.randint(0, 10, size=(np.int(popln*0.5) ,2))
    cross_vec=[]
    for i in np.arange(len(random)):
        C1 = Best_population[random[i][0]]
        C2 = Best_population[random[i][1]]
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < Pc):
            z = r[r < Pc]
            r.tolist(); z.tolist()
            v= np.where(r==z[0])
            w= v[1][0]
            Ori_C1 = C1.copy()
            C1[w+1:] = C2[w+1:]
            C2[w+1:] = Ori_C1[w+1:]
            cross_vec.append(C1)
            cross_vec.append(C2)
        else:
            C1=C1 ; C2=C2
            cross_vec.append(C1)
            cross_vec.append(C2)
    crossover = copy.deepcopy(cross_vec)
#%%    
    #   Mutation
    for i in np.arange(popln):
        r = np.random.uniform(0,1,size=(1,bits))
        if np.any(r < Pm):
            z = r[r < Pm]
            r.tolist(); z.tolist()
            v= np.where(r==z[0])
            w= v[1][0]
            if cross_vec[i][w] == 1:
                cross_vec[i][w] = 0
            else:
                cross_vec[i][w] = 1
        else:   
            cross_vec[i]=cross_vec[i]
    cv=cross_vec # now mutated popln became new initial popln
    
    m_acc =[] ; m_FIT =[]
    m_indx = [] ; m_D=[]
    for i in np.arange(len(cv)): 
        m_ind = list(np.where(cv[i]==1))
        m_indx.append(m_ind)
        m_I = m_indx[0][0]
        
        m_test = data[data.columns[list(m_I)]]
        Xm_train,Xm_test,ym_train,ym_test = train_test_split(m_test,y,test_size = 0.2,random_state = 42)
        m_model = RandomForestClassifier()
        m_model.fit(Xm_train,ym_train)
        ym_pred = m_model.predict(Xm_test)
        mm_acc = accuracy_score(ym_test,ym_pred)
        m_acc.append(mm_acc) 
        m_c = np.linspace(0.2,1,5)
        m_fit=[]
        for j in np.arange(len(m_c)):     
            m_fitnes = mm_acc - (j*len(m_I) / len(data.columns))
            m_fit.append(m_fitnes)
            m_F = np.max(m_fit)
        m_FIT.append(m_F)
    BEST_ACC = np.max(m_FIT)    
    best_acc_ind = np.argmax(m_FIT)
    T_ind = m_indx[best_acc_ind]
    attri.append(T_ind)
    BEST_ACC_vec.append(BEST_ACC)
    ACC_vec.append(best_acc_ind)

Best_attri_no = np.argmax(BEST_ACC_vec)

Best_attri = attri[Best_attri_no]
ATR = data[data.columns[list(Best_attri[0])]]
ATTRIBUTS = ATR.columns

print("BEST ATTRIBUTS ARE  - \n\n %s\n\n"%list(ATTRIBUTS))
#%%