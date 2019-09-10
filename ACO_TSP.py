import numpy as np
import pandas as pd
import random
import math
import time

##############
def dist_route(route,dist_df):
    return(sum([D_df[route[i]].loc[route[i+1]] for i in range(len(route)-1)]))
#############

###############
def city_selection(starting_city,PC_mat,city_left,city_visited):
    while(True):
        #print(count)
        if len(city_left)==1:
            city_visited.append(city_visited[0])
            #print(city_visited)
            break
        #print(PC_mat)
        #print(starting_city)
        if np.random.uniform()<q:
            city_visited.append(PC_mat[starting_city].idxmax())
            city_left.remove(starting_city)
            PC_mat=PC_mat.drop(starting_city)
            starting_city=city_visited[-1]
            #print("direct",starting_city)

            #print(city_left)

            #print(j)
        else:
            prob_array=PC_mat[str(starting_city)]/PC_mat[str(starting_city)].sum() #probabiltic value for each city from the city where the an is now
            #print(prob_array)
            #print(city_left)
            city_visited.append(np.random.choice(city_left,p=prob_array))
            city_left.remove(starting_city)
            PC_mat=PC_mat.drop(starting_city)
            starting_city=city_visited[-1]
            #print("with probab",starting_city)

        #print(city_left)
            #print(j)
        #print(city_visited)
    return city_visited
###############

################
def update_pheromone(routes,dis_route,D_df,P):
    P=P-0.01
    df=pd.DataFrame(routes)
    #df['Dis']=dis_route
    print(df)
    min_idx=np.where(dis_route==dis_route.min())[0]
    max_idx=np.where(dis_route==dis_route.max())[0]

    minimum_dist_routes=df.iloc[min_idx[0]]
    if len(minimum_dist_routes)>1:
        choose_route=np.random.choice(min_idx[0])
        min_route=minimum_dist_routes.loc[choose_route]

    else:
        min_route=minimum_dist_routes
        print(min_route)
        print(len(min_route))
    #print(len(min_route))

    for k in range(len(min_route)-1):
        temp=P[min_route[k]].loc[min_route[k+1]]
        print(temp)
        print("Hello")
        P[min_route[k]].loc[min_route[i+1]]=temp+0.1
    print(P)

    return P




################



#def pheromone_update_function():


num_ants=10                                                                     #choose number of ants
num_nodes=6                                                                     #choose number of cities
cities=[chr(x) for x in range(97,97+num_nodes)]

#make a distance matrix among the cities

#t0=time.time()
dist_vals=np.random.randint(10,100,num_nodes**2)
dist_mat=np.reshape(dist_vals,[num_nodes,num_nodes])
D = np.tril(dist_mat) + np.tril(dist_mat, -1).T
D_df=pd.DataFrame(D,index=cities, columns=cities)
visibility_mat=1/D
np.fill_diagonal(visibility_mat, 0)

P=pd.DataFrame(np.ones([num_nodes,num_nodes]),index=cities, columns=cities)

while(True):

    PC_mat_orig=P*visibility_mat
    PC_mat_orig=pd.DataFrame(PC_mat_orig,index=cities, columns=cities)



    q=0.3
    dis_route=pd.Series(range(0,num_ants))
    routes=[]
    for i in range(num_ants):
        city_visited=[]
        city_left=[chr(x) for x in range(97,97+num_nodes)]
        #print(city_left)
        PC_mat=PC_mat_orig
        #PC_mat=pd.DataFrame(PC_mat,index=city_left, columns=city_left)
        #print(PC_mat)
        starting_city=np.random.choice(city_left)
        city_visited.append(starting_city)
        count=1
        route=city_selection(starting_city,PC_mat,city_left,city_visited)
        #print (route)
        routes.append(route)
        dis_route[i]=dist_route(route,D_df)
        #print(dis_route)
    #route_df=pd.DataFrame(routes)
    P=update_pheromone(routes,dis_route,D_df,P)























#t1=time.time()
#print(t1-t0)

#apply loop for each of the ants
# distt=np.zeros([num_nodes,num_nodes])
# t0=time.time()
# count=1
# for i in range(num_nodes):
#     for j in range(num_nodes):
#         if i==j:
#             distt[i][j]==0
#             #count=count+1
#         else:
#             choose=np.random.randint(10,100,1)
#             distt[i][j]=choose
#             distt[j][i]=choose
#             #count=count+2
#     #if count==num_nodes**2:
#     #    break
#
#
# t1=time.time()
# print(t1-t0)
