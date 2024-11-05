# %% Imports
from itertools import permutations
import pandas
# from hmmlearn import hmm
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import scipy.stats as stats
# import seaborn as sns
import pandas as pd
from scipy import interpolate
from numpy import var, mean, sqrt
import pickle
# from edfio import Edf, EdfSignal
from datetime import datetime as dt
import random
import copy
from sklearn.cluster import KMeans
from scipy.stats import ranksums
from sklearn import linear_model
from scipy import stats
from scipy.linalg import norm
from matplotlib.pyplot import imshow
import scipy.io
from numpy.linalg import eig
import networkx as nx
from matplotlib.ticker import MultipleLocator
import time as time1
import mne
from hclinic_mne import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

#%% R-squared and HCS probability computing

pat = ['01','02','03','11','12','14','15','16','17','18']

t_ant = time1.time()

n_states = 12
r2 = np.zeros((n_states,len(pat)), dtype=np.float32)
n2 = 12*20 # number of samples for 12h

# State probability
p_dbs = np.zeros((n2, n_states, len(pat)), dtype=np.float32) # day before seizure
p_sd = np.zeros((n2, n_states, len(pat)), dtype=np.float32) # seizure day

for z, patient in enumerate(pat):
    
    # read previous parameters
    EC_ar_dbs = np.load('ID'+patient+'/EC_c_ar.npy')
    EC_ar_sd = np.load('ID'+patient+'/EC_p_ar.npy')
    fc_ar_dbs = np.load('ID'+patient+'/fc_c_ar.npy')
    fc_ar_sd = np.load('ID'+patient+'/fc_p_ar.npy')
    N_ar = np.load('ID'+patient+'/N_ar.npy')
    H_p = np.load('ID'+patient+'/H_p.npy')
    H_c = np.load('ID'+patient+'/H_c.npy')
    
    N = len(N_ar)
    Nc_all = len(EC_ar_dbs)
    
    # concatenate day before the seizure and seizure day
    EC_sd_dbs = np.zeros((Nc_all, np.sum(N_ar)*2), dtype=np.float32)
    fc_sd_dbs = np.zeros((np.sum(N_ar)*2), dtype=np.float32)
    
    t_ar = 0
    for i in range(N):
        EC_sd_dbs[:, t_ar:t_ar+N_ar[i]] = EC_ar_dbs[:, :N_ar[i], i]
        EC_sd_dbs[:, t_ar+np.sum(N_ar):t_ar+np.sum(N_ar) +
                  N_ar[i]] = EC_ar_sd[:, :N_ar[i], i]
        fc_sd_dbs[t_ar:t_ar+N_ar[i]] = fc_ar_dbs[:N_ar[i], i]
        fc_sd_dbs[t_ar+np.sum(N_ar):t_ar+np.sum(N_ar)+N_ar[i]
                  ] = fc_ar_sd[:N_ar[i], i]
        t_ar += N_ar[i]
    
    # Sample data
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
    centers = np.mean(centers1, axis=1)

    cent_ord = list(reversed(sorted(centers)))
    cent_ord = np.array(cent_ord)
    
    # to obtain the indexes
    ind_cent = np.zeros((12))

    for i in range(12):
        ind_cent[i] = np.array(np.where(centers == cent_ord[i]))

    n = np.size(labels)
    labels_ind = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for j in range(12):
            if labels[i] == ind_cent[j]:
                labels_ind[i] = j

    # % sort by mean connectivity ##################################################################
    fc_centers = np.zeros(12)
    for i in range(12):
        c1 = np.array(np.where(labels_ind == i))
        c1 = c1.flatten()
        n = len(c1)
        temp = 0
        for j in range(n):
            temp += fc_sd_dbs[c1[j]]
            # temp += np.mean(fc_sd_dbs[:,c1[j]])
        fc_centers[i] = temp/n

    fc_cent_ord = list(reversed(sorted(fc_centers)))
    fc_cent_ord = np.array(fc_cent_ord)

    ind_fc_cent = np.zeros((12))

    for i in range(12):
        ind_fc_cent[i] = np.array(np.where(fc_centers == fc_cent_ord[i]))

    n = np.size(labels_ind)
    fc_labels_ind = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for j in range(12):
            if labels_ind[i] == ind_fc_cent[j]:
                fc_labels_ind[i] = j
    
    # state probability
    for k in range(n_states):
        t_ar = 0
        for i in range(n2):
            for j in range(N_ar[i]):
                # l = labels_ind[j+t_ar]
                l = fc_labels_ind[j+t_ar]
                if l == k:
                    p_dbs[i, k, z] += 1
            p_dbs[i, k, z] /= N_ar[i]
            t_ar += N_ar[i]

        # t_ar = int(np.size(labels_ind)/2)
        t_ar = int(np.size(fc_labels_ind)/2)
        for i in range(n2):
            for j in range(N_ar[i]):
                # l = labels_ind[j+t_ar]
                l = fc_labels_ind[j+t_ar]
                if l == k:
                    p_sd[i, k, z] += 1
            p_sd[i, k, z] /= N_ar[i]
            t_ar += N_ar[i]
            
    # r2 computing####################################################################
    p_sd_dbs_res = p_sd[:,:,z] - p_dbs[:,:,z]
    H_res = H_p - H_c

    h = 8 # for 8h 

    for i in range(n_states):

        x = p_sd_dbs_res[n2-20*h:, i].reshape((-1, 1))
        y = H_res[n2-20*h:]

        model = LinearRegression()
        model.fit(x, y)
        coef = model.coef_
        intercept = model.intercept_

        r2[i,z] = model.score(x, y)
        
t_act = (time1.time() - t_ant)/60
print("{:.2f}".format(t_act)+' min')