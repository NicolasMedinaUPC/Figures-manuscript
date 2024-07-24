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

#%% R-squared HCS probability with power, heart rate, mean connectivity and mean centrality

h = 8 # number of hours
# select parameter
x1 = np.array([pw_dbs[300-20*h:],pw_sd[300-20*h:]]) # power
# x1 = np.array([hr_dbs[300-20*h:],hr_sd[300-20*h:]]) # heart rate
# x1 = np.array([fc_dbs[200-20*h:],fc_sd[200-20*h:]]) # connectivity
# x1 = np.array([cent_dbs[300-20*h:],cent_sd[300-20*h:]]) # centrality
y1 = np.array([p_dbs[300-20*h:,0],p_sd[300-20*h:,0]]) # HCS probability

# Bootstrapping
n_days = 2 # number of days
w1 = 5 # 15 minutes time window
r2_1 = np.zeros((n_days,int(h*20/w1)), dtype=np.float32)

for i in range(n_days):
    for j in range(int(h*20/w1)):
        x = x1[i,j*w1:(j+1)*w1].reshape((-1, 1))
        y = y1[i,j*w1:(j+1)*w1].reshape((-1, 1))
        model = LinearRegression()
        model.fit(x, y)
        r2_1[i,j] = model.score(x, y) # R-squared
r2_dbs_mean = np.mean(r2_1[0,:])
r2_sd_mean = np.mean(r2_1[1,:])
r2_dbs_std = np.std(r2_1[0,:])
r2_sd_std = np.std(r2_1[1,:])

day = [i+1 for i in range(n_days)]
fig, ax = plt.subplots()
ax.errorbar(2, r2_sd_mean, r2_sd_std, fmt='o', linewidth=3, capsize=6, color='red', label='Seizure day')
ax.errorbar(1, r2_dbs_mean, r2_dbs_std, fmt='o', linewidth=3, capsize=6, color='blue', label='Day before seizure')
ax.set_title("R\u00b2 [HCS probability - Power] \n("+str(h)+" hours before seizure)")
# ax.set_title("R\u00b2 [HCS probability - Heart rate] \n("+str(h)+" hours before seizure)")
# ax.set_title("R\u00b2 [HCS probability - Functional connectivity] \n("+str(h)+" hours before seizure)")
# ax.set_title("R\u00b2 [HCS probability - Centrality] \n("+str(h)+" hours before seizure)")
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_ylabel("")
ax.set_ylabel("Coefficient of determination [R\u00b2]")
ax.yaxis.set_tick_params()
ax.set_xticklabels(['','','Day before seizure','Seizure day'])

#%% Spearman correlation between HCS probability and other variable

h=9.5 # number of hours
N=30 # time window samples 
n_corr = np.size(p_HCS_sd)-N+1 # correlation samples
Spearman_corr_sd = np.zeros(n_corr)
Spearman_corr_dbs = np.zeros(n_corr)
Spearman_pvalue_sd = np.zeros(n_corr)
Spearman_pvalue_dbs = np.zeros(n_corr)


for i in range(n_corr):
    # power
    corr_matrix = stats.spearmanr(p_sd[i:i+N],pw_sd[i:i+N]) 
    Spearman_corr_sd[i] = corr_matrix[0]
    Spearman_pvalue_sd[i] = corr_matrix[1]
    corr_matrix = stats.spearmanr(p_dbs[i:i+N],pw_dbs[i:i+N]) 
    Spearman_corr_dbs[i] = corr_matrix[0]
    Spearman_pvalue_dbs[i] = corr_matrix[1]
    
    # heart rate
    # corr_matrix = stats.spearmanr(p_sd[i:i+N],hr_sd[i:i+N]) 
    # Spearman_corr_sd[i] = corr_matrix[0]
    # Spearman_pvalue_sd[i] = corr_matrix[1]
    # corr_matrix = stats.spearmanr(p_dbs[i:i+N],hr_dbs[i:i+N]) 
    # Spearman_corr_dbs[i] = corr_matrix[0]
    # Spearman_pvalue_dbs[i] = corr_matrix[1]
    
    # connectivity
    # corr_matrix = stats.spearmanr(p_sd[i:i+N],fc_sd[i:i+N]) 
    # Spearman_corr_sd[i] = corr_matrix[0]
    # Spearman_pvalue_sd[i] = corr_matrix[1]
    # corr_matrix = stats.spearmanr(p_dbs[i:i+N],fc_dbs[i:i+N]) 
    # Spearman_corr_dbs[i] = corr_matrix[0]
    # Spearman_pvalue_dbs[i] = corr_matrix[1]
    
    # centrality
    # corr_matrix = stats.spearmanr(p_sd[i:i+N],cent_sd[i:i+N]) 
    # Spearman_corr_sd[i] = corr_matrix[0]
    # Spearman_pvalue_sd[i] = corr_matrix[1]
    # corr_matrix = stats.spearmanr(p_dbs[i:i+N],cent_dbs[i:i+N]) 
    # Spearman_corr_dbs[i] = corr_matrix[0]
    # Spearman_pvalue_dbs[i] = corr_matrix[1]

# moving average
window = 5
average_sd = []
average_dbs = []
for ind in range(len(Spearman_corr_sd) - window + 1):
    average_sd.append(np.mean(Spearman_corr_sd[ind:ind+window]))
    average_dbs.append(np.mean(Spearman_corr_dbs[ind:ind+window]))
      
hrs = 9.5 - 1.5
horas=[-(i*hrs)/np.size(average_sd) for i in range(np.size(average_sd))]
horas=list(reversed(horas))
# Number of points after interpolation
num_points = len(average_sd)*4
x_new = np.linspace(min(horas), max(horas), num_points)
av_sd = np.interp(x_new, horas, average_sd)
av_dbs = np.interp(x_new, horas, average_dbs)
fig, ax2= plt.subplots(figsize=(40, 24))
ax2.set_title('Spearman correlation [HCS probability - Gamma power band]\n',fontsize=70)
# ax2.set_title('Spearman correlation [HCS probability - Heart rate]\n',fontsize=70)
# ax2.set_title('Spearman correlation [HCS probability - Functional connectivity]\n',fontsize=70)
# ax2.set_title('Spearman correlation [HCS probability - Centrality]\n',fontsize=70)
plt.legend(loc=4,prop={'size': 30})
ax2.set_xlim([np.min(horas),np.max(horas)])
ax2.yaxis.set_tick_params(labelsize=30)
ax2.set_xlabel("\ntime(h), \u0394h=180s",fontsize=60)
ax2.xaxis.set_tick_params(labelsize=60)
ax2.yaxis.set_tick_params(labelsize=60)
plt.axhline(y=0.36,linewidth=4,color='k',label='Significance threshold (p<0.05)')
plt.axhline(y=-0.36,linewidth=4,color='k')
plt.legend(loc=4,prop={'size': 50})
plt.ylim(ymax = 1, ymin = -1)
