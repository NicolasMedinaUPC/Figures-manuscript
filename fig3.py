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

#%% Computing centraly entropy, eigevectors centrality for each reference

# Read EDF
filename='patient1_sz_day.EDF'
f = mne.io.read_raw_edf(filename)

# Define initial conditions
horas = 15 # 15h preictal
N = 300 # samples to compute centrality entropy
num_step_H = int((60/(0.6*N/60))*horas) # number of iterations to compute centrality entropy

fs_ini=f.info['sfreq'] # sampling frequency
signalLabels=f.ch_names # channels labels
selectallContacts='''
FSAL : 1-7,
FSPL : 1-10,
PL : 1-8,
FMAL : 1-10,
FMML : 1-15,
FMPL : 1-15,
FBL : 1-15,
IMSL : 1-8,
IAL : 1-8,
PTL : 1-10,
HAL : 1-8,10-15,
HML : 1-15,
TSL : 1-5,
SML : 1-18,
FBR : 1-13,15,
'''
selectContactsIndices_all=SelectContacts(signalLabels,selectallContacts) # contacts to analize
Nc_all=len(selectContactsIndices_all) # number of channels
readcontacts_all=[selectContactsIndices_all[i] for i in range(Nc_all)] # contacts to read

############################################################
eventon_text='01:41:56' # seizure onset time
############################################################
eventon=time(eventon_text)

# Seizure day
f = f_p2_d3
filestart_ini=f.info['meas_date'] # recording onset time
filestart_ini=dt.combine(filestart_ini.date(),filestart_ini.time())
filestart=dt(filestart_ini.year,filestart_ini.month,filestart_ini.day+1)
filestart=dt.combine(filestart.date(),filestart_ini.time())
eventend=dt.combine(filestart.date(),eventon.time())
eventendn=(eventend-filestart_ini).total_seconds()
n_end1 = int((eventendn+1)*fs_ini)
n_start1 = n_end1-int((0.6*N)*fs_ini+2*fs_ini)

# Day before seizure
f = f_p2_d2
filestart_ini=f.info['meas_date'] # recording onset time
filestart_ini=dt.combine(filestart_ini.date(),filestart_ini.time())
filestart=dt(filestart_ini.year,filestart_ini.month,filestart_ini.day+1)
filestart=dt.combine(filestart.date(),filestart_ini.time())
eventend=dt.combine(filestart.date(),eventon.time())
eventendn=(eventend-filestart_ini).total_seconds()
n_end2 = int((eventendn+1)*fs_ini)
n_start2 = n_end2-int((0.6*N)*fs_ini+2*fs_ini)

# Downsampling
fsold=fs_ini
fs=512
q=int(fsold/fs)
print(f'---> Downsampling from {fsold} to {fs} Hz...')

# Cut raw signal  
n_e = int(0.6*N*fs+fs)
n_s = int(fs)

# Filter properties
Nf=3 # order of the filter
fll,fhh=[1, 150]
sos = signal.butter(Nf, [fll,fhh], btype='bandpass',fs=fs, output='sos')
f0=list(range(50,250+1,50))
rf=35

# Pearson correlation parameters
number_seconds=0.6;
window=number_seconds*fs
time_samples=fs*0.6*N;
num_steps_EC=int(time_samples/window)

# Remove artefacts parameters
th=3
th2=3
vec2 = np.zeros((Nc_all),dtype=np.float64)
mean_conn = np.zeros((num_steps_EC),dtype=np.float64)
std_conn = np.zeros((num_steps_EC),dtype=np.float64)
mean_energy = np.zeros((num_steps_EC),dtype=np.float64)
std_energy = np.zeros((num_steps_EC),dtype=np.float64)

# Centrality entropy
n = Nc_all # number of samples
H_c = np.zeros((num_step_H),dtype=np.float32) # day before seizure
H_p = np.zeros((num_step_H),dtype=np.float32) # seizure day

# Eigenvectors centrality
EC_ar_dbs = np.zeros((Nc_all,num_steps_EC,num_step_H),dtype=np.float32) # day before seizure
EC_ar_sd = np.zeros((Nc_all,num_steps_EC,num_step_H),dtype=np.float32) # seizure day

# Functional connectivity
func_conn_ar_dbs = np.zeros((num_steps_EC,num_step_H),dtype=np.float32) # day before seizure
func_conn_ar_sd = np.zeros((num_steps_EC,num_step_H),dtype=np.float32) # seizure day

# channels of each electrodo for bipolar reference
electrode_range = np.array([7,10,8,10,15,15,15,8,8,10,14,15,5,18,14])

for i in range(num_step_H):
    
    ##################################################################
################## pre-seizure period - seizure day
    ##################################################################
    filename='patient1_sz_day.EDF'
    f = mne.io.read_raw_edf(filename)
    
    ### monopolar reference
    data_all=f.get_data(readcontacts_all,n_start1,n_end1)*1e6
    
    ### common average
    data_all=f.get_data(readcontacts_all,n_start1,n_end1)*1e6
    reference_mean=np.mean(data_all,axis=0)
    data_all=data_all-reference_mean
    
    ### bipolar reference
    data_all=f.get_data(readcontacts_all,n_start1,n_end1)*1e6
    data_bipolar = np.zeros((len(data_all)-len(electrode_range),np.size(data_all,1)))
    m = 0
    for i2 in electrode_range:
        for j2 in range(i2-1):
            data_bipolar[m+j2,:] = data_all[m+j2,:]-data_all[m+j2+1,:]
        m += j2+1
    data_all = data_bipolar
    
    # Downsampling
    data=signal.decimate(data_all,q,axis=-1)
    Nt=data.shape[-1]
    y = data    

    # Band-pass 1 to 150Hz, Remove slow drifts and aliasing
    y = signal.sosfiltfilt(sos, y, axis=-1)

    # Remove AC noise (50 Hz and multiples)    
    y=notchfiltfilt(y,f0,fs,rf=rf) ### check if rf is ok
    y=notchfiltfilt(y,f0,fs,rf=rf)
    y=y.astype(np.float32)
    y=y[:,n_s:n_e]
    
    # Eigenvector centrality
    for j in range(num_steps_EC):        
        y_600ms = y[:,int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)]
        corr_matrix = np.corrcoef(y_600ms)    
        corr_matrix = abs(corr_matrix)
                   
        # Remove artefacts    
        # COMPUTE MEAN CONNECTIVITY (over all pairs)
        Mat = corr_matrix
        Mat1 = np.triu(Mat, k=1)
        vec_aux = Mat1.flatten('F')
        vec_aux = vec_aux.compress((vec_aux!=0).flat)
        # FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
        vec_aux_FT=(1/2)*np.log((1+vec_aux)/(1-vec_aux))
        vec_aux_FT_m=np.mean(vec_aux_FT)
        # ANTI-TRANSFORM
        vec_aux_mean=(np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)       
        mean_conn[j] = vec_aux_mean
        std_conn[j] = np.std(vec_aux_FT, ddof=1)
        #COMPUTE MEAN ENERGY
        Data1=y_600ms
        for kk in range (Nc_all):
            vec2[kk]=(norm(Data1[kk,:]))*(norm(Data1[kk,:]))/window
        mean_energy[j] = np.median(vec2)
        std_energy[j] = np.std(vec2, ddof=1)
        
        # Connectivity with artefacts
        func_conn_wa_sd[j,num_step_H-i-1] = vec_aux_mean
        
        # Centrality with artefacts
        G = nx.from_numpy_array(corr_matrix)
        centrality_dict = nx.eigenvector_centrality(G, weight='weight')
        centrality_list = [float(x) for x in list(centrality_dict.values())]
        EC_wa_sd[:,j,num_step_H-i-1] = np.array(centrality_list,dtype=float)
       
    me=mean_energy;
    mc=mean_conn;
    se=std_energy;
    sc=std_conn;
    #%ENERGY%%%%%%%%%%%%%%%%%
    med_me=np.median(np.log(me));
    std_me=abs(np.std(np.log(me), ddof=1));       
    med_se=np.median(np.log(se));
    std_se=abs(np.std(np.log(se), ddof=1));
    #%CONNECTIVITY%%%%%%%%%
    med_mc=np.median(mc);
    std_mc=np.std(mc, ddof=1);
    med_sc=np.median(sc);
    std_sc=np.std(sc, ddof=1);
    
    #total condition
    A_me=(np.log(me)<=med_me+th*std_me) & (np.log(me)>=med_me-th*std_me);
    B_me=(np.log(se)<=med_se+th2*std_se) & (np.log(se)>=med_se-th2*std_se);
    ind_me=np.argwhere(A_me & B_me); 
    A_mc=(mc<=med_mc+th*std_mc) & (mc>=med_mc-th*std_mc);
    B_mc=(sc<=med_sc+th2*std_sc) & (sc>=med_sc-th2*std_sc);
    ind_mc=np.argwhere(A_mc & B_mc); 
    ind_rem_artifact_p=np.intersect1d(ind_me,ind_mc);
        
    ##################################################################
########## control period - day before seizure
    ##################################################################   
    filename='patient1_day_before_sz.EDF'
    f = mne.io.read_raw_edf(filename)
    
    ### monopolar reference
    data_all=f.get_data(readcontacts_all,n_start2,n_end2)*1e6
    
    ### common average
    data_all=f.get_data(readcontacts_all,n_start2,n_end2)*1e6
    reference_mean=np.mean(data_all,axis=0)
    data_all=data_all-reference_mean
    
    ### bipolar reference
    data_all=f.get_data(readcontacts_all,n_start2,n_end2)*1e6
    data_bipolar = np.zeros((len(data_all)-len(electrode_range),np.size(data_all,1)))
    m = 0
    for i2 in electrode_range:
        for j2 in range(i2-1):
            data_bipolar[m+j2,:] = data_all[m+j2,:]-data_all[m+j2+1,:]
        m += j2+1
    data_all = data_bipolar
       
    # Downsampling
    data=signal.decimate(data_all,q,axis=-1)
    Nt=data.shape[-1]
    y = data    

    # Band-pass 1 to 150Hz, Remove slow drifts and aliasing
    y = signal.sosfiltfilt(sos, y, axis=-1)

    # Remove AC noise (50 Hz and multiples)    
    y=notchfiltfilt(y,f0,fs,rf=rf)
    y=notchfiltfilt(y,f0,fs,rf=rf)
    y=y.astype(np.float32)
    y=y[:,n_s:n_e]
    
    ###### Eigenvector centrality
    for j in range(num_steps_EC):        
        y_600ms = y[:,int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)]
        corr_matrix = np.corrcoef(y_600ms)    
        corr_matrix = abs(corr_matrix)
               
        # Remove artefacts    
        # COMPUTE MEAN CONNECTIVITY (over all pairs)
        Mat = corr_matrix
        Mat1 = np.triu(Mat, k=1)
        vec_aux = Mat1.flatten('F')
        vec_aux = vec_aux.compress((vec_aux!=0).flat)
        #FISHER TRANSFORM - quita la condicion de conectividad 0 a 1
        vec_aux_FT=(1/2)*np.log((1+vec_aux)/(1-vec_aux))
        vec_aux_FT_m=np.mean(vec_aux_FT)
        #ANTI-TRANSFORM
        vec_aux_mean=(np.exp(2*vec_aux_FT_m)-1)/(np.exp(2*vec_aux_FT_m)+1)     
        mean_conn[j] = vec_aux_mean
        std_conn[j] = np.std(vec_aux_FT, ddof=1)
        #COMPUTE MEAN ENERGY
        Data1=y_600ms
        for kk in range (Nc_all):
            vec2[kk]=(norm(Data1[kk,:]))*(norm(Data1[kk,:]))/window
        mean_energy[j] = np.median(vec2)
        std_energy[j] = np.std(vec2, ddof=1)   
        
        # Connectivity with artefacts
        func_conn_wa_dbs[j,num_step_H-i-1] = vec_aux_mean
        
        # Centrality with artefacts
        G = nx.from_numpy_array(corr_matrix)
        centrality_dict = nx.eigenvector_centrality(G, weight='weight')
        centrality_list = [float(x) for x in list(centrality_dict.values())]
        EC_wa_dbs[:,j,num_step_H-i-1] = np.array(centrality_list,dtype=float)
       
    me=mean_energy;
    mc=mean_conn;
    se=std_energy;
    sc=std_conn;
    #%ENERGY%%%%%%%%%%%%%%%%%
    med_me=np.median(np.log(me));
    std_me=abs(np.std(np.log(me), ddof=1));       
    med_se=np.median(np.log(se));
    std_se=abs(np.std(np.log(se), ddof=1));      
    #%CONNECTIVITY%%%%%%%%%
    med_mc=np.median(mc);
    std_mc=np.std(mc, ddof=1);
    med_sc=np.median(sc);
    std_sc=np.std(sc, ddof=1);
    #total condition
    A_me=(np.log(me)<=med_me+th*std_me) & (np.log(me)>=med_me-th*std_me);
    B_me=(np.log(se)<=med_se+th2*std_se) & (np.log(se)>=med_se-th2*std_se);
    ind_me=np.argwhere(A_me & B_me); 
    A_mc=(mc<=med_mc+th*std_mc) & (mc>=med_mc-th*std_mc);
    B_mc=(sc<=med_sc+th2*std_sc) & (sc>=med_sc-th2*std_sc);
    ind_mc=np.argwhere(A_mc & B_mc); 
    ind_rem_artifact_c=np.intersect1d(ind_me,ind_mc);
    
    ### Functional connectivity without artefacts
    ind_rem_artifact = np.intersect1d(ind_rem_artifact_p,ind_rem_artifact_c);
    N_ar[num_step_H-i-1] = np.size(ind_rem_artifact)
    func_conn_ar_dbs[:N_ar[num_step_H-i-1],num_step_H-i-1] = func_conn_wa_dbs[ind_rem_artifact,num_step_H-i-1]
    func_conn_ar_sd[:N_ar[num_step_H-i-1],num_step_H-i-1] = func_conn_wa_sd[ind_rem_artifact,num_step_H-i-1]
    
    ### Eigenvector centrality without artefacts
    EC_ar_dbs[:,:N_ar[num_step_H-i-1],num_step_H-i-1] = EC_d2_15h_N300[:,ind_rem_artifact,num_step_H-i-1]
    EC_ar_sd[:,:N_ar[num_step_H-i-1],num_step_H-i-1] = EC_d3_15h_N300[:,ind_rem_artifact,num_step_H-i-1]
        
    # new time window
    n_end1 = n_start1+int(2*fs_ini)
    n_start1 = n_end1-int(0.6*N*fs_ini+2*fs_ini)
    n_end2 = n_start2+int(2*fs_ini)
    n_start2 = n_end2-int(0.6*N*fs_ini+2*fs_ini)
    
#%% Clustering network states using K-means algorithm

### Cluster by centrality
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
