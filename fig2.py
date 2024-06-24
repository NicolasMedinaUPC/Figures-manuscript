#%% Define libraries

import numpy as np
import matplotlib.pyplot as plt
from hclinic_mne import *
import mne
from matplotlib.ticker import MultipleLocator
import networkx as nx
from numpy.linalg import eig
import scipy.io
from matplotlib.pyplot import imshow
from scipy.linalg import norm
from scipy import stats
import pandas 
from sklearn import linear_model
from scipy.stats import ranksums
from sklearn.cluster import KMeans
import copy
import random
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression

#%% Clustering network states using K-means algorithm

### Clustering by centrality
N = len(N_ar)
Nc_all = len(EC_ar_dbs)
EC_sd_dbs = np.zeros((Nc_all,np.sum(N_ar)*2),dtype=np.float32)
fc_sd_dbs = np.zeros((np.sum(N_ar)*2),dtype=np.float32)
# concatenate both days
t_ar = 0
for i in range(N):
    EC_sd_dbs[:,t_ar:t_ar+N_ar[i]] = EC_ar_dbs[:,:N_ar[i],i]
    EC_sd_dbs[:,t_ar+np.sum(N_ar):t_ar+np.sum(N_ar)+N_ar[i]] = EC_ar_sd[:,:N_ar[i],i]
    fc_sd_dbs[t_ar:t_ar+N_ar[i]] = fc_ar_dbs[:N_ar[i],i]
    fc_sd_dbs[t_ar+np.sum(N_ar):t_ar+np.sum(N_ar)+N_ar[i]] = fc_ar_sd[:N_ar[i],i]
    t_ar += N_ar[i]
# k-means
data = EC_sd_dbs.T
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
# indexes
ind_cent = np.zeros((12))
for i in range (12):
    ind_cent[i] = np.array(np.where(centers == cent_ord[i]))
# sort labels by centrality    
n = np.size(labels)
labels_ind = np.zeros(n, dtype=np.int32)
for i in range (n):
    for j in range (12):
        if labels[i] == ind_cent[j]:
            labels_ind[i] = j

### sort by mean connectivity
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
            
#%% Computing HCS probability

n_states = 12
n = 300
p_dbs = np.zeros((n,n_states), dtype=np.float32)
p_sd = np.zeros((n,n_states), dtype=np.float32)
for k in range(n_states):
    # day before seizure
    t_ar = 0
    for i in range (n):
        for j in range (N_ar[i]):
            l = labels_ind[j+t_ar]
            if l == k:
                p_dbs[i,k] += 1
        p_dbs[i,k] /= N_ar[i] # probability
        t_ar += N_ar[i]
    # seizure day
    t_ar = int(np.size(labels_ind)/2)
    for i in range (n):
        for j in range (N_ar[i]):
            l = labels_ind[j+t_ar]
            if l == k:
                p_sd[i,k] += 1 # probability
        p_sd[i,k] /= N_ar[i]
        t_ar += N_ar[i]

#%% R-squared of each state to explain entropy drop

n_states = 12
p_sd_dbs_res = p_sd - p_dbs # probability
H_res = H_p - H_c # centrality entropy
state = [i+1 for i in range(n_states)] # states
ve = np.zeros((n_states), dtype=np.float32) # R-squared
h = 8 # number of hours
for i in range(n_states):
    x = p_sd_dbs_res[300-20*h:,i].reshape((-1, 1))
    y = H_res[300-20*h:]
    model = LinearRegression()
    model.fit(x, y)
    coef = model.coef_
    intercept = model.intercept_    
    ve[i] = model.score(x, y)
    
#%% Statistical analysis of HCS probability

minutes = 15
N = int(minutes/3)
n = int(300/N)
p_value = np.zeros(n)
statistic = np.zeros(n)
for k in range(n):
    sample1 = p_dbs[k*N:(k+1)*N,0]
    sample2 = p_sd[k*N:(k+1)*N,0]
    # Compute the observed test statistic
    statistic[k], p_value[k] = ranksums(sample1, sample2)
###clustering
cluster = np.array(np.where(p_value<0.05)).flatten()
n2 = np.size(cluster)
t1 = 1
for i in range(n2-1):
    if cluster[i] != cluster[i+1]-1:
        t1 += 1
sz_label = np.ones(t1)
label = np.zeros(t1)
k = 0
for i in range(n2-1):
    if cluster[i] == cluster[i+1]-1:
        sz_label[k] += 1       
    if cluster[i] != cluster[i+1]-1 and i < n2-2:
        label[k] = cluster[i]
        k += 1
    if i == n2-2:
        label[k] = cluster[i]
label[t1-1] = cluster[n2-1]