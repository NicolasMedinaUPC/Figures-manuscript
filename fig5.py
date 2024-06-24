#%% Define libraries

import numpy as np
import time as time1
import copy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy.linalg import eig
from itertools import permutations
from sklearn.cluster import KMeans

#%% low-dimensional statistical network model
ord1 = 4
min3 = 20*(2)
it_n = 300*(min3)
step = 0.05
err1 = 0.01
err2 = 0.0005
# define mean connectivity
thr1, thr2 = 0.20-err1, 0.20+err1
# thr1, thr2 = 0.25-err1, 0.25+err1
# define mean centrality
thr_cent1, thr_cent2 = 0.4825-err2, 0.4825+err2 # 
# thr_cent1, thr_cent2 = 0.4835-err2, 0.4835+err2 # 
nn = 300
conn_mean = np.zeros(min3)
centrality = np.zeros(min3)
cmm_t = np.zeros((ord1,ord1,it_n))
cent = np.zeros(nn*min3,dtype=np.float32)

for a in range(min3):
    va_ac = 0
    cent_ac = 0
    while ((va_ac < thr1 or va_ac > thr2)): # connectivity condition 
        va_ac = 0
        cent_ac = 0
        for b in range(nn):
            cent = 0
            while ((cent < thr_cent1 or cent > thr_cent2)): # centrality condition 
                cm1 = np.ones((ord1,ord1))
                n = len(cm1)        
                eigv_pos = False
                while(eigv_pos==False):
                    for i in range(n):
                        for j in range(n):                            
                            # 0.2->factor=0.245, 0.25->factor=0.32
                            # if i != j: cm1[i,j] = np.random.randn()*0.245
                            if i != j: cm1[i,j] = np.random.randn()*0.32                                               
                            if j < i: cm1[i,j] = cm1[j,i]           
                    Eigenvalue, Eigenvector = eig(cm1)
                    eigv_pos = all(Eigenvalue>=0)
                #COMPUTE MEAN CONNECTIVITY (over all pairs)
                Mat = abs(cm1)
                Mat1 = np.triu(Mat, k=1)
                vec_aux = Mat1.flatten('F')
                vec_aux = vec_aux.compress((vec_aux!=0).flat)        
                for v1 in range(len(vec_aux)):
                    if vec_aux[v1] > 0.99: vec_aux[v1] = 0.99        
                #FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
                vec_aux_FT=(1/2)*np.log((1+vec_aux)/(1-vec_aux))
                vec_aux_FT_m=np.mean(vec_aux_FT)
                #ANTI-TRANSFORM
                vec_aux_mean=(np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)                         
                #centrality
                Mat = abs(cm1)   
                G = nx.from_numpy_array(Mat)
                centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
                centrality_list = [float(x) for x in list(centrality_dict.values())]
                cent = np.mean(np.array(centrality_list,dtype=float))
            cent_ac += cent            
            cmm_t[:,:,300*a+b] = cm1 # covariance matrix finded
            va_ac += vec_aux_mean        
        va_ac /= nn
        cent_ac /= nn    
    conn_mean[a] = va_ac
    centrality[a] = cent_ac

#%% join connectivity and centrality
cm_dbs = np.concatenate((cm_dbs1,cm_dbs2),axis=2) # covariance matrices calculated previously
cm_sd = np.concatenate((cm_sd1,cm_sd2),axis=2) # covariance matrices calculated previously
n = np.size(cm_dbs,axis=2)
conn_mean_dbs = np.zeros(n)
conn_mean_sd = np.zeros(n)
cent_mean_dbs = np.zeros(n)
cent_mean_sd = np.zeros(n)
c_mean_dbs = np.zeros(20*4)
c_mean_sd = np.zeros(20*4)
cent_dbs = np.zeros(20*4)
cent_sd = np.zeros(20*4)
it_n = int(300*20*4)
EC_dbs = np.zeros((it_n,ord1))
EC_sd = np.zeros((it_n,ord1))

for i in range(n):    
    
    ### Day before seizure
    #COMPUTE MEAN CONNECTIVITY (over all pairs)
    Mat = abs(cm_dbs[:,:,i])
    Mat1 = np.triu(Mat, k=1)
    vec_aux = Mat1.flatten('F')
    vec_aux = vec_aux.compress((vec_aux!=0).flat)        
    for v1 in range(len(vec_aux)):
        if vec_aux[v1] > 0.99: vec_aux[v1] = 0.99        
    #FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
    vec_aux_FT=(1/2)*np.log((1+vec_aux)/(1-vec_aux))
    vec_aux_FT_m=np.mean(vec_aux_FT)
    #ANTI-TRANSFORM
    vec_aux_mean=(np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)
    conn_mean_dbs[i] = vec_aux_mean
    #centrality  
    G = nx.from_numpy_array(Mat)
    centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
    centrality_list = [float(x) for x in list(centrality_dict.values())]
    cent_mean_dbs[i] = np.mean(np.array(centrality_list,dtype=float))
    EC_dbs[i,:] = np.array(centrality_list,dtype=float)
    
    ### Seizure day
    #COMPUTE MEAN CONNECTIVITY (over all pairs)
    Mat = abs(cm_sd[:,:,i])
    Mat1 = np.triu(Mat, k=1)
    vec_aux = Mat1.flatten('F')
    vec_aux = vec_aux.compress((vec_aux!=0).flat)        
    for v1 in range(len(vec_aux)):
        if vec_aux[v1] > 0.99: vec_aux[v1] = 0.99        
    #FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
    vec_aux_FT=(1/2)*np.log((1+vec_aux)/(1-vec_aux))
    vec_aux_FT_m=np.mean(vec_aux_FT)
    #ANTI-TRANSFORM
    vec_aux_mean=(np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)
    conn_mean_sd[i] = vec_aux_mean
    #centrality  
    G = nx.from_numpy_array(Mat)
    centrality_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
    centrality_list = [float(x) for x in list(centrality_dict.values())]
    cent_mean_sd[i] = np.mean(np.array(centrality_list,dtype=float))
    EC_sd[i,:] = np.array(centrality_list,dtype=float)

for i in range(20*4):
    c_mean_dbs[i] = np.mean(conn_mean_dbs[300*i:300*(i+1)])
    c_mean_sd[i] = np.mean(conn_mean_sd[300*i:300*(i+1)]) 
    cent_dbs[i] = np.mean(cent_mean_dbs[300*i:300*(i+1)])
    cent_sd[i] = np.mean(cent_mean_sd[300*i:300*(i+1)]) 

#%% Clustering network states using K-means algorithm

### Clustering by centrality
EC_sd_dbs = np.zeros((4*60*100*2,4),dtype=np.float32)
for i in range(4*60*100*2):
    EC_sd_dbs[:4*60*100,:] = EC_dbs
    EC_sd_dbs[4*60*100:,:] = EC_sd
# Sample data
data = EC_sd_dbs
# Define the number of clusters
num_clusters = 12
# Create a KMeans object and fit the data
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)
# Get the cluster labels for each data point
labels = kmeans.labels_
# Get the cluster centers
centers1 = kmeans.cluster_centers_
centers = np.mean(centers1,axis=1)
cent_ord = list(reversed(sorted(centers)))
cent_ord = np.array(cent_ord)
ind_cent = np.zeros((12))
for i in range (12):
    ind_cent[i] = np.array(np.where(centers == cent_ord[i]))  
n = np.size(labels)
labels_ind = np.zeros(n, dtype=np.int32)
for i in range (n):
    for j in range (12):
        if labels[i] == ind_cent[j]:
            labels_ind[i] = j
            
### sort by mean connectivity
fc_sd_dbs = np.concatenate((cent_mean_dbs,cent_mean_sd))
fc_centers = np.zeros(12)
for i in range (12):
    c1 = np.array(np.where(labels_ind == i))
    c1 = c1.flatten()
    n = len(c1)
    temp = 0
    for j in range(n):
        temp += fc_sd_dbs[c1[j]]
    fc_centers[i] = temp/n    
fc_cent_ord = list(reversed(sorted(fc_centers)))
fc_cent_ord = np.array(fc_cent_ord)
ind_fc_cent = np.zeros((12))
for i in range (12):
    ind_fc_cent[i] = np.array(np.where(fc_centers == fc_cent_ord[i]))   
n = np.size(labels_ind)
fc_labels_ind = np.zeros(n, dtype=np.int32)
for i in range (n):
    for j in range (12):
        if labels_ind[i] == ind_fc_cent[j]:
            fc_labels_ind[i] = j

#%% computing HCS probability
n_states = 12
N = 300
n = int(len(fc_labels_ind)/2/N)
p_dbs = np.zeros((n,n_states), dtype=np.float32)
p_sd = np.zeros((n,n_states), dtype=np.float32)

for k in range(n_states):
    # day before seizure
    t_ar = 0
    for i in range (n):
        for j in range (N):
            l = fc_labels_ind[j+t_ar]
            if l == k:
                p_dbs[i,k] += 1
        p_dbs[i,k] /= N
        t_ar += N
    # seizure day    
    t_ar = int(np.size(fc_labels_ind)/2)
    for i in range (n):
        for j in range (N):
            l = fc_labels_ind[j+t_ar]
            if l == k:
                p_sd[i,k] += 1
        p_sd[i,k] /= N
        t_ar += N

#%% mean connectivity and mean centrality for each state
n_states = 12
N = 300
n = int(len(fc_labels_ind)/2/N)
# connectivity
c_dbs_sd = np.concatenate((conn_mean_dbs,conn_mean_sd))
c_dbs = np.zeros((n,n_states), dtype=np.float32)
c_sd = np.zeros((n,n_states), dtype=np.float32)
# centrality
cent_dbs_sd = np.concatenate((cent_mean_dbs,cent_mean_sd))
cent_dbs = np.zeros((n,n_states), dtype=np.float32)
cent_sd = np.zeros((n,n_states), dtype=np.float32)
for k in range(n_states):
    # day before seizure
    t_ar = 0
    for i in range (n):
        temp1 = 0
        for j in range (N):
            l = fc_labels_ind[j+t_ar]
            if l == k:
                c_dbs[i,k] += c_dbs_sd[j+t_ar]
                cent_dbs[i,k] += cent_dbs_sd[j+t_ar]
                temp1 += 1
        c_dbs[i,k] /= temp1
        cent_dbs[i,k] /= temp1
        t_ar += N
    # seizure day
    t_ar = int(np.size(fc_labels_ind)/2)
    for i in range (n):
        temp1 = 0
        for j in range (N):
            l = fc_labels_ind[j+t_ar]
            if l == k:
                c_sd[i,k] += c_dbs_sd[j+t_ar]
                cent_sd[i,k] += cent_dbs_sd[j+t_ar]
                temp1 += 1
        c_sd[i,k] /= temp1
        cent_sd[i,k] /= temp1
        t_ar += N

