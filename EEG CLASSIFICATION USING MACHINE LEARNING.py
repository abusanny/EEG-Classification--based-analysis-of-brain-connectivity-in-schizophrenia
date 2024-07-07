#!/usr/bin/env python
# coding: utf-8

# In[2]:


from glob import glob
import os


# In[3]:


pip install mne


# In[4]:


import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Loading the full paths of all the matching files

# In[5]:


all_files_path=glob('dataverse_files/*.edf')


#   creating healthy and diseased files from the rest

# In[6]:


healthy_file_path=[i for i in all_files_path if 'h' in i.split('/')[1]]
patient_file_path=[i for i in all_files_path if 's' in i.split('/')[1]]


# preparing the data for further analysis and feature extraction

# In[7]:


len(all_files_path)


# In[8]:


print(len(patient_file_path),len(healthy_file_path))


# Read and preprocess EEG data

# In[9]:


def read_data(file_path):
    data=mne.io.read_raw_edf(file_path,preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=0.5,h_freq=45)
    epochs=mne.make_fixed_length_epochs(data,duration=5,overlap=1)
    array=epochs.get_data()
    return array


# Capture output silently

# In[10]:


get_ipython().run_cell_magic('capture', '', 'control_epoch_array=[read_data(i) for i in healthy_file_path]\npatient_epoch_array=[read_data(i) for i in patient_file_path]\n')


# Create labels for control and patient data

# In[11]:


control_epoch_label=[len(i)*[0] for i in control_epoch_array]
patient_epoch_label=[len(i)*[1] for i in patient_epoch_array]


# In[12]:


len(control_epoch_label),len(patient_epoch_label)


# Combine data and labels

# In[13]:


data_list=control_epoch_array+patient_epoch_array
label_list=control_epoch_label+patient_epoch_label


#  Create group list for each epoch

# In[14]:


group_list=[[i]*len(j) for i,j in enumerate(data_list)]
len(group_list)


# Create final arrays

# In[15]:


data_array=np.vstack(data_list)
label_array=np.hstack(label_list)
group_array=np.hstack(group_list)


#  Print shapes of the final arrays

# In[16]:


print(data_array.shape, label_array.shape, group_array.shape)


# In[17]:


from scipy import stats
def mean(x):
    return np.mean(x,axis=-1)
def std(x):
    return np.std(x,axis=-1)
def ptp(x):
    return np.ptp(x,axis=-1)
def var(x):
    return np.var(x,axis=-1)
def minim(x):
    return np.min(x,axis=-1)
def maxim(x):
    return np.max(x,axis=-1)
def argmin(x):
    return np.argmin(x,axis=-1)
def argmax(x):
    return np.argmax(x,axis=-1)
def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))
def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)
def skewness(x):
    return stats.skew(x,axis=-1)
def kurtosis(x):
    return stats.kurtosis(x,axis=-1)
def concatenate_features(x):
    return np.concatenate((mean(x),std(x),ptp(x),var(x),minim(x),maxim(x),
                           argmin(x),argmax(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x)),axis=-1)


# In[18]:


features=[]
for d in data_array:
    features.append(concatenate_features(d))


# In[19]:


features_array=np.array(features)
features_array.shape


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold


# Define the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Define the parameter grid
param_grid = {'clf__C': [0.1, 0.5, 0.7, 1, 3, 5, 7]}

# Define the cross-validator
gkf = GroupKFold(n_splits=5)

# Initialize GridSearchCV with a scoring method
gscv = GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=12, scoring='accuracy')

# Fit the model
gscv.fit(features_array, label_array, groups=group_array)


# In[24]:


gscv.best_score_


# In[70]:





# In[71]:





# In[ ]:




