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

#%% Computing centraly entropy, mean connectivity and mean centrality

# Read EDF
filename='patient1_sz_day.EDF'
f = mne.io.read_raw_edf(filename)

# Define initial conditions
horas = 15 # 15h preictal
N = 300 # samples to compute entropy
num_step_H = int((60/(0.6*N/60))*horas)

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
# Nx600 ms time window with 1 second boundary condition
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
# Nx600 ms time window with 1 second boundary condition
n_end2 = int((eventendn+1)*fs_ini)
n_start2 = n_end2-int((0.6*N)*fs_ini+2*fs_ini)

# Downsampling
fsold=fs_ini
fs=512
q=int(fsold/fs)
print(f'---> Downsampling from {fsold} to {fs} Hz...')

# Cut raw signal to get Nx600ms time window
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

# Mean centrality
mean_cent_dbs = np.zeros((num_step_H),dtype=np.float32) # day before seizure
mean_cent_sd = np.zeros((num_step_H),dtype=np.float32) # seizure day

# Functional connectivity
func_conn_ar_dbs = np.zeros((num_steps_EC,num_step_H),dtype=np.float32) # day before seizure
func_conn_ar_sd = np.zeros((num_steps_EC,num_step_H),dtype=np.float32) # seizure day

# Mean connectivity
mean_fc_dbs = np.zeros((num_step_H),dtype=np.float32) # day before seizure
mean_fc_sd = np.zeros((num_step_H),dtype=np.float32) # seizure day

# Iterations to calculate centrality entropy
for i in range(num_step_H):
    
    ##################################################################
################## pre-seizure period - seizure day
    ##################################################################
    filename='patient1_sz_day.EDF'
    f = mne.io.read_raw_edf(filename)
    data_all=f.get_data(readcontacts_all,n_start1,n_end1)*1e6
    
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
    
    # iterations to calculate eigenvectors centrality every 600 ms
    for j in range(num_steps_EC):        
        y_600ms = y[:,int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)] # read 600ms of signal
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
    data_all=f.get_data(readcontacts_all,n_start2,n_end2)*1e6
       
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
    
    ###### # iterations to calculate eigenvectors centrality every 600 ms
    for j in range(num_steps_EC):        
        y_600ms = y[:,int(fs*j*number_seconds):int(fs*(j+1)*number_seconds)] # read 600ms of signal
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
    
    ### Centrality entropy
    covMAT = np.cov(EC_c)
    Dv,V = eig(covMAT)
    H_c[num_step_H-i-1] = (n/2)*(1+np.log(2*np.pi))+0.5*np.sum(np.log(Dv)) # day before seizure
    covMAT = np.cov(EC_p)
    Dv,V = eig(covMAT)
    H_p[num_step_H-i-1] = (n/2)*(1+np.log(2*np.pi))+0.5*np.sum(np.log(Dv)) # seizure day
    
    ### Mean connectivity
    mean_fc_dbs[num_step_H-i-1] = np.mean(func_conn_ar_dbs[:,num_step_H-i-1]) # day before seizure
    mean_fc_sd[num_step_H-i-1] = np.mean(func_conn_ar_sd[:,num_step_H-i-1]) # seizure day
    
    ### Mean centrality
    mean_cent_dbs[num_step_H-i-1] = np.mean(EC_ar_dbs[:,:,num_step_H-i-1]) # day before seizure
    mean_cent_sd[num_step_H-i-1] = np.mean(EC_ar_sd[:,:,num_step_H-i-1]) # seizure day
    
    # new time window
    n_end1 = n_start1+int(2*fs_ini)
    n_start1 = n_end1-int(0.6*N*fs_ini+2*fs_ini)
    n_end2 = n_start2+int(2*fs_ini)
    n_start2 = n_end2-int(0.6*N*fs_ini+2*fs_ini)
    
#%% Computing power and heart rate

# Read EDF
filename='patient1_day_before_sz.EDF'
f = mne.io.read_raw_edf(filename)

signalLabels=f.ch_names
i=0
for label in signalLabels:
    if label == 'EKG L': signalLabels[i] = 'EKG L1'
    if label == 'EKG R': signalLabels[i] = 'EKG R1'
    i+=1
selectallContacts_power='''
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
selectContactsIndices_power=SelectContacts(signalLabels,selectallContacts_power)

### Power baseline
# initial conditions
event_baseline='08:30:00'
eventbaseline=time(event_baseline)
eventbas=dt.combine(filestart_ini.date(),eventbaseline.time())
eventbasn=(eventbas-filestart_ini).total_seconds()
fs_ini=f.info['sfreq']
Nc_all=len(selectContactsIndices_all)
# define 5 minutes baseline time window
n_end_bas = int((eventbasn)*fs_ini)
n_start_bas = n_end_bas-int(60*5*fs_ini)
channellabels=[]
fsold=fs_ini
fs=512
q=int(fsold/fs)
# read EDFraw data
readcontacts_all=[selectContactsIndices_power[i] for i in range(Nc_all)]
data_bas=f.get_data(readcontacts_all,n_start_bas,n_end_bas)*1e6
channellabels=[signalLabels[j] for j in readcontacts_all]
data=data_bas
# common average
reference_mean=np.mean(data,axis=0)
data=data-reference_mean
# Downsampling
data=signal.decimate(data,q,axis=-1)
Nt=data.shape[-1]
# Remove AC noise (50 Hz and multiples)
x=data
y=(x.T-x.mean(axis=-1)).T
y=signal.detrend(y,axis=-1)
f0=list(range(50,200+1,50))
rf=35
y=notchfiltfilt(y,f0,fs,rf=rf) ### check if rf is ok
y=y.astype(np.float32)

# select band
freqrange = [0.08, 0.4] #ultra slow
# freqrange = [1, 4] #delta
# freqrange = [4, 8] #theta
# freqrange = [8, 12] #alpha
# freqrange = [12, 30] #beta
# freqrange = [30, 80] #gamma
# freqrange = [80, 150] #HFO

Pxx_sum = np.zeros((Nc_all),dtype=np.float32)
NFFT = 1024
for i in range(Nc_all): 
    frequencies, psd = signal.welch(y[i,:], fs, nperseg=NFFT) # power spectral density (Pxx
    freq_indices = np.where((frequencies >= freqrange[0]) & (frequencies <= freqrange[1]))
    Pxx_sum[i] = np.trapz(psd[freq_indices], frequencies[freq_indices]) # integrate Pxx
Pxx_sum_mean=np.mean(Pxx_sum)
Pxx_sum_desv=np.std(Pxx_sum)

### Instantaneous power

# initial conditions
horas = 15
minutes = 60*horas
# select day
filename='patient1_sz_day.EDF'
# filename='patient1_day_before_sz.EDF'
f = mne.io.read_raw_edf(filename)
fs_ini = f.info['sfreq']
Nc_all=len(selectContactsIndices_all)
############################################################
eventon_text='01:41:56' # seizure onset time
############################################################
eventon=time(eventon_text)
filestart=dt(filestart_ini.year,filestart_ini.month,filestart_ini.day+1)
filestart=dt.combine(filestart.date(),filestart_ini.time())
eventend=dt.combine(filestart.date(),eventon.time())
eventendn=(eventend-filestart_ini).total_seconds()
# 60 seconds time window with 0.5 second boundary condition
n_end = int((eventendn+0.5)*fs_ini)
n_start = n_end-int(61*fs_ini)
fsold=fs_ini
fs=512
q=int(fsold/fs)
print(f'---> Downsampling from {fsold} to {fs} Hz...')
# 60 seconds time window
n_e = int(60.5*fs)
n_s = int(0.5*fs)
readcontacts_all=[selectContactsIndices_power[i] for i in range(Nc_all)]

zscore = np.zeros((Nc_all,minutes),dtype=np.float32)
for k in range(minutes):   
    channellabels=[]
    data_all=f.get_data(readcontacts_all,n_start,n_end)*1e6
    readcontacts=readcontacts_all
    channellabels=[signalLabels[j] for j in readcontacts]
    data=data_all
    # common average
    reference_mean=np.mean(data_all,axis=0)
    data=data-reference_mean
    # Downsampling
    data=signal.decimate(data,q,axis=-1)
    # Remove AC noise (50 Hz and multiples)
    x=data
    y=(x.T-x.mean(axis=-1)).T
    y=signal.detrend(y,axis=-1)
    f0=list(range(50,200+1,50))
    rf=35
    y=notchfiltfilt(y,f0,fs,rf=rf) ### check if rf is ok
    y=y.astype(np.float32)
    fs = int(fs)    
    for i in range(Nc_all): 
        frequencies, psd = signal.welch(y[i,:], fs, nperseg=NFFT) # power spectral density (Pxx)
        freq_indices = np.where((frequencies >= freqrange[0]) & (frequencies <= freqrange[1]))
        Pxx_sum = np.trapz(psd[freq_indices], frequencies[freq_indices]) #integrate Pxx
        # zscore values
        zscore[i,minutes-k-1] = (Pxx_sum-Pxx_sum_mean)/Pxx_sum_desv        
    n_end = n_start+int(fs_ini)
    n_start = n_end-int(61*fs_ini)

### EKG
# initial conditions
selectallContacts_ecg='''
EKG L: 1
'''
selectContactsIndices_ecg=SelectContacts(signalLabels,selectallContacts_ecg)
Nc_ecg=len(selectContactsIndices_ecg)
n_end = int((eventendn+0.5)*fs_ini)
n_start = n_end-int(61*fs_ini)
fsold=fs_ini
fs=512
q=int(fsold/fs)
# 60 seconds time window
n_e = int(60.5*fs)
n_s = int(0.5*fs)
readcontacts_ecg=[selectContactsIndices_ecg[i] for i in range(Nc_ecg)]

# read 60 seconds of EKG signal
ecg = np.zeros((minutes*60*fs),dtype=np.float32)
for k in range(minutes):
    # select day
    filename='patient1_sz_day.EDF'
    # filename='patient1_day_before_sz.EDF'
    f = mne.io.read_raw_edf(filename)
    data_ecg=f.get_data(readcontacts_ecg,n_start,n_end)*1e6
    # Downsampling
    data_ecg = signal.decimate(data_ecg,q,axis=-1)
    data_ecg = data_ecg[:,n_s:n_e]
    ecg[k*60*fs:(k+1)*60*fs] = data_ecg        
    n_end = n_start+int(fs_ini)
    n_start = n_end-int(61*fs_ini)
    
# Band pass filter
rtol=0.05 # extra tolerance at either side of the window to avoid -3dB at specified freqs
N=4
# fll,fhh=[0.2,40]
fll,fhh=[60,90] # band pass filter
y = ecg 
# Remove AC noise (50 Hz and multiples)
x=y
y=(x.T-x.mean(axis=-1)).T
y=signal.detrend(y,axis=-1)
f0=list(range(50,200+1,50))
rf=35
y=notchfiltfilt(y,f0,fs,rf=rf) ### check if rf is ok
y=y.astype(np.float32)
sos = signal.butter(N, [fll*(1-rtol),fhh*(1+rtol)], btype='bandpass',fs=fs, output='sos')
ecg_f = signal.sosfiltfilt(sos, y, axis=-1)

# computing heart rate (count R-picks per minute)
picks = np.zeros((minutes),dtype=np.float32)
for k in range(minutes):
    c_picks = 0
    j = 0
    ecg_f_deriv = np.gradient(ecg_f[fs*60*k+180:fs*60*(k+1)-180])
    ecg_f_deriv2 = np.power(ecg_f_deriv,2)    
    i_act = 0
    i_ant = 0
    i_act_acum = 0
    # remove artifacts
    for i in range(np.size(ecg_f_deriv2)):
        if ecg_f_deriv2[i] > 5*np.mean(ecg_f_deriv2) and i>j:
            c_picks += 1
            j = i + 256 # do not consider peaks less than 0.5 seconds
            i_act = i - i_ant
            i_act = (i_act/fs)*60
            i_act_acum += i_act
            i_ant = i
    if c_picks != 0:
        i_act_acum /= c_picks
    else: i_act_acum = 0
    picks[k] = int(i_act_acum)
