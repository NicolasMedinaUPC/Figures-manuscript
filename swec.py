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

# define the patient
patient = "ID01"
mat_file = "info/"+patient+"_info.mat"
mat = scipy.io.loadmat(mat_file)

# find the seizure onset instant
fs = mat["fs"].item()
# n_ch = np.size(mat["EEG"],0)
sz_onset = mat["seizure_begin"][0].item()
hr_onset = math.ceil(sz_onset/3600)
res = round(sz_onset%3600*fs)

mat_file = patient+"/"+patient+"_"+str(hr_onset)+"h.mat"
mat = scipy.io.loadmat(mat_file)
len_mat = np.size(mat["EEG"],1)
n_ch = np.size(mat["EEG"],0)
del mat

total_horas = 12
N = 300
num_step_H = int(60/(0.6*N/60))
# num_step_H = 1

n_e = int(0.6*N*fs+fs)
n_s = int(fs)

# Filter properties
Nf = 3  # order of the filter
fll, fhh = [1, 150]
sos = signal.butter(Nf, [fll, fhh], btype='bandpass', fs=fs, output='sos')
f0 = list(range(50, 250+1, 50))
rf = 35

# Pearson correlation
number_seconds = 0.6
window = number_seconds*fs
time_samples = fs*0.6*N
num_steps_EC = int(time_samples/window)

# Remove artefacts
th = 3
th2 = 3
vec2 = np.zeros((n_ch), dtype=np.float64)
mean_conn = np.zeros((num_steps_EC), dtype=np.float64)
std_conn = np.zeros((num_steps_EC), dtype=np.float64)
mean_energy = np.zeros((num_steps_EC), dtype=np.float64)
std_energy = np.zeros((num_steps_EC), dtype=np.float64)

# eigenvectors centrality
EC_p_wa = np.zeros((n_ch,num_steps_EC, num_step_H*total_horas), dtype=np.float32) # pre-seizure period with artifact
EC_c_wa = np.zeros((n_ch,num_steps_EC, num_step_H*total_horas), dtype=np.float32) # control period with artifact
EC_p_ar = np.zeros((n_ch,num_steps_EC, num_step_H*total_horas), dtype=np.float32) # pre-seizure period without artifact
EC_c_ar = np.zeros((n_ch,num_steps_EC, num_step_H*total_horas), dtype=np.float32) # control period without artifact

# centrality entropy
H_p = np.zeros((num_step_H*total_horas), dtype=np.float32) # pre-seizure period
H_c = np.zeros((num_step_H*total_horas), dtype=np.float32) # control period
N_ar = np.zeros((num_step_H*total_horas),dtype=np.int32) # samples with artifact removal

# mean functional connectivity
fc_p_wa = np.zeros((num_steps_EC, num_step_H*total_horas), dtype=np.float32) # pre-seizure period with artifact
fc_c_wa = np.zeros((num_steps_EC, num_step_H*total_horas), dtype=np.float32) # control period with artifact
fc_p_ar = np.zeros((num_steps_EC, num_step_H*total_horas), dtype=np.float32) # pre-seizure period without artifact
fc_c_ar = np.zeros((num_steps_EC, num_step_H*total_horas), dtype=np.float32) # control period without artifact

t_ant = time1.time()
for horas in range(total_horas):
    n_end = fs*3600+2*fs
    n_start = n_end-int((0.6*N)*fs+2*fs)

    # seizure day
    mat_file = patient+"/"+patient+"_"+str(hr_onset-horas)+"h.mat"
    mat_sd = scipy.io.loadmat(mat_file)
    # n_ch = np.size(mat_sd["EEG"],0)
    data_sd = np.zeros((n_ch,fs*3600+2*fs))
    
    data_sd[:,-int(res+fs):] = mat_sd["EEG"][:,:int(res+fs)]    
    mat_file = patient+"/"+patient+"_"+str(hr_onset-horas-1)+"h.mat"
    mat_sd = scipy.io.loadmat(mat_file)    
    data_sd[:,:-int(res+fs)] = mat_sd["EEG"][:,int(res-fs):]
    
    # day before seizure
    mat_file = patient+"/"+patient+"_"+str(hr_onset-horas-24)+"h.mat"
    mat_dbs = scipy.io.loadmat(mat_file)
    data_dbs = np.zeros((n_ch,fs*3600+2*fs))
       
    data_dbs[:,-int(res+fs):] = mat_dbs["EEG"][:,:int(res+fs)]    
    mat_file = patient+"/"+patient+"_"+str(hr_onset-horas-1-24)+"h.mat"
    mat_dbs = scipy.io.loadmat(mat_file)    
    data_dbs[:,:-int(res+fs)] = mat_dbs["EEG"][:,int(res-fs):]
        
    del mat_sd, mat_dbs
    
    # H computing
    for i in range(num_step_H):

        print(i+1+horas*num_step_H, end=' '),

        ##################################################################
    # pre-seizure period
        ##################################################################

        # data_all = f_p2_d3.get_data(readcontacts_all, n_start1, n_end1)
        data = data_sd[:,n_start:n_end]

        # referenciado con la media
        reference_median = np.median(data, axis=0)
        data = data-reference_median

        # Band-pass 1 to 150Hz, Remove slow drifts and aliasing
        y = signal.sosfiltfilt(sos, data, axis=-1)
        del data

        # Remove AC noise (50 Hz and multiples)
        y = notchfiltfilt(y, f0, fs, rf=rf)  # check if rf is ok
        y = notchfiltfilt(y, f0, fs, rf=rf)
        y = y.astype(np.float32)
        y = y[:, n_s:n_e]

        # Eigenvector centrality
        for j in range(num_steps_EC):

            y_600ms = y[:, int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)]
            corr_matrix = np.corrcoef(y_600ms)
            corr_matrix = abs(corr_matrix)

            # str_wa_sd[:, :, j] = corr_matrix
            
            G = nx.from_numpy_array(corr_matrix)
            centrality_dict = nx.eigenvector_centrality(G, weight='weight')
            centrality_list = [float(x) for x in list(centrality_dict.values())]
            EC_p_wa[:,j,num_step_H*total_horas-i-1-horas*num_step_H] = np.array(centrality_list,dtype=float)

        # Remove artefacts

            # COMPUTE MEAN CONNECTIVITY (over all pairs)
            Mat = corr_matrix
            Mat1 = np.triu(Mat, k=1)
            vec_aux = Mat1.flatten('F')
            vec_aux = vec_aux.compress((vec_aux != 0).flat)

            # FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
            vec_aux_FT = (1/2)*np.log((1+vec_aux)/(1-vec_aux))
            vec_aux_FT_m = np.mean(vec_aux_FT)
            # ANTI-TRANSFORM
            vec_aux_mean = (np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)
            
            fc_p_wa[j,num_step_H*total_horas-i-1-horas*num_step_H] = vec_aux_mean

            mean_conn[j] = vec_aux_mean
            std_conn[j] = np.std(vec_aux_FT, ddof=1)

            # COMPUTE MEAN ENERGY
            Data1 = y_600ms

            for kk in range(n_ch):
                vec2[kk] = (norm(Data1[kk, :]))*(norm(Data1[kk, :]))/window

            mean_energy[j] = np.median(vec2)
            std_energy[j] = np.std(vec2, ddof=1)

        me = mean_energy
        mc = mean_conn
        se = std_energy
        sc = std_conn

        # %ENERGY%%%%%%%%%%%%%%%%%
        med_me = np.median(np.log(me))
        std_me = abs(np.std(np.log(me), ddof=1))

        med_se = np.median(np.log(se))
        std_se = abs(np.std(np.log(se), ddof=1))

        # %CONNECTIVITY%%%%%%%%%
        med_mc = np.median(mc)
        std_mc = np.std(mc, ddof=1)

        med_sc = np.median(sc)
        std_sc = np.std(sc, ddof=1)

        # total condition
        A_me = (np.log(me) <= med_me+th*std_me) & (np.log(me) >= med_me-th*std_me)
        B_me = (np.log(se) <= med_se+th2*std_se) & (np.log(se) >= med_se-th2*std_se)
        ind_me = np.argwhere(A_me & B_me)
        A_mc = (mc <= med_mc+th*std_mc) & (mc >= med_mc-th*std_mc)
        B_mc = (sc <= med_sc+th2*std_sc) & (sc >= med_sc-th2*std_sc)
        ind_mc = np.argwhere(A_mc & B_mc)
        ind_rem_artifact_p = np.intersect1d(ind_me, ind_mc)

        ##################################################################
    # control period
        ##################################################################

        # data_all = f_p2_d2.get_data(readcontacts_all, n_start2, n_end2)*1e6
        data = data_dbs[:,n_start:n_end]

        # referenciado con la media
        reference_median = np.median(data, axis=0)
        data = data-reference_median

        # Band-pass 1 to 150Hz, Remove slow drifts and aliasing
        y = signal.sosfiltfilt(sos, data, axis=-1)
        del data

        # Remove AC noise (50 Hz and multiples)
        y = notchfiltfilt(y, f0, fs, rf=rf)  # check if rf is ok
        y = notchfiltfilt(y, f0, fs, rf=rf)
        y = y.astype(np.float32)
        y = y[:, n_s:n_e]

        # Eigenvector centrality
        for j in range(num_steps_EC):

            y_600ms = y[:, int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)]
            corr_matrix = np.corrcoef(y_600ms)
            corr_matrix = abs(corr_matrix)

            # str_wa_dbs[:, :, j, num_step_H-i-1] = corr_matrix
            
            G = nx.from_numpy_array(corr_matrix)
            centrality_dict = nx.eigenvector_centrality(G, weight='weight')
            centrality_list = [float(x) for x in list(centrality_dict.values())]
            EC_c_wa[:,j,num_step_H*total_horas-i-1-horas*num_step_H] = np.array(centrality_list,dtype=float)

        # Remove artefacts

            # COMPUTE MEAN CONNECTIVITY (over all pairs)
            Mat = corr_matrix
            Mat1 = np.triu(Mat, k=1)
            vec_aux = Mat1.flatten('F')
            vec_aux = vec_aux.compress((vec_aux != 0).flat)
            
            fc_c_wa[j,num_step_H*total_horas-i-1-horas*num_step_H] = vec_aux_mean

            # FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
            vec_aux_FT = (1/2)*np.log((1+vec_aux)/(1-vec_aux))
            vec_aux_FT_m = np.mean(vec_aux_FT)
            # ANTI-TRANSFORM
            vec_aux_mean = (np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)

            mean_conn[j] = vec_aux_mean
            std_conn[j] = np.std(vec_aux_FT, ddof=1)

            # COMPUTE MEAN ENERGY
            Data1 = y_600ms

            for kk in range(n_ch):
                vec2[kk] = (norm(Data1[kk, :]))*(norm(Data1[kk, :]))/window

            mean_energy[j] = np.median(vec2)
            std_energy[j] = np.std(vec2, ddof=1)

        me = mean_energy
        mc = mean_conn
        se = std_energy
        sc = std_conn

        # %ENERGY%%%%%%%%%%%%%%%%%
        med_me = np.median(np.log(me))
        std_me = abs(np.std(np.log(me), ddof=1))

        med_se = np.median(np.log(se))
        std_se = abs(np.std(np.log(se), ddof=1))

        # %CONNECTIVITY%%%%%%%%%
        med_mc = np.median(mc)
        std_mc = np.std(mc, ddof=1)

        med_sc = np.median(sc)
        std_sc = np.std(sc, ddof=1)

        # total condition
        A_me = (np.log(me) <= med_me+th*std_me) & (np.log(me) >= med_me-th*std_me)
        B_me = (np.log(se) <= med_se+th2*std_se) & (np.log(se) >= med_se-th2*std_se)
        ind_me = np.argwhere(A_me & B_me)
        A_mc = (mc <= med_mc+th*std_mc) & (mc >= med_mc-th*std_mc)
        B_mc = (sc <= med_sc+th2*std_sc) & (sc >= med_sc-th2*std_sc)
        ind_mc = np.argwhere(A_mc & B_mc)
        ind_rem_artifact_c = np.intersect1d(ind_me, ind_mc)

        ind_rem_artifact = np.intersect1d(ind_rem_artifact_p, ind_rem_artifact_c)
        N_ar[num_step_H*total_horas-i-1-horas*num_step_H] = np.size(ind_rem_artifact)

        # eigenvector centrality without artefacts
        EC_p_ar[:,:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],
                num_step_H*total_horas-i-1-horas*num_step_H] = EC_p_wa[:, ind_rem_artifact, num_step_H*total_horas-i-1-horas*num_step_H]
        EC_c_ar[:,:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],
                num_step_H*total_horas-i-1-horas*num_step_H] = EC_c_wa[:, ind_rem_artifact, num_step_H*total_horas-i-1-horas*num_step_H]
        
        # functional connectivity
        fc_p_ar[:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],
                num_step_H*total_horas-i-1-horas*num_step_H] = fc_p_wa[ind_rem_artifact, num_step_H*total_horas-i-1-horas*num_step_H]
        fc_c_ar[:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],
                num_step_H*total_horas-i-1-horas*num_step_H] = fc_c_wa[ind_rem_artifact, num_step_H*total_horas-i-1-horas*num_step_H]

        ##### Centrality entropy - Pre-seizure period
        covMAT = np.cov(EC_p_ar[:,:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],num_step_H*total_horas-i-1-horas*num_step_H])
        Dv,V = eig(covMAT)
        H_p[num_step_H*total_horas-i-1-horas*num_step_H] = (n_ch/2)*(1+np.log(2*np.pi))+0.5*np.sum(np.log(Dv))

        ##### Centrality entropy - Control period
        covMAT = np.cov(EC_c_ar[:,:N_ar[num_step_H*total_horas-i-1-horas*num_step_H],num_step_H*total_horas-i-1-horas*num_step_H])
        Dv,V = eig(covMAT)
        H_c[num_step_H*total_horas-i-1-horas*num_step_H] = (n_ch/2)*(1+np.log(2*np.pi))+0.5*np.sum(np.log(Dv))

        n_end = n_start+int(2*fs)
        n_start = n_end-int(0.6*N*fs+2*fs)

t_act = (time1.time() - t_ant)/60
print("{:.2f}".format(t_act)+' min')