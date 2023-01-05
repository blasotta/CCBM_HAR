import numpy as np
import pandas as pd
import torch
import os
from motionSense import set_data_types, creat_time_series, make_views, get_windows
from dataframeSplitter import DataFrameSplitter

output_dir = '../../data/MotionSense'

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

# select sensor data types, typically all are wanted so set them all
# attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
# print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS [0:6]
# print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
# print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    


# print("[INFO] -- Splitting into train, val, test") 
dfs = DataFrameSplitter(method="subject")
subject_col = "id"
# These are the subjects for the train and validation set, the subjects not in
# this list will be used to build the test set, e.g. in this case
# subject ids 19-23 are in the test set (Subject-Independent split)
split_subjects = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0]

# These are the subjects for the training set, from the split set above it
# follows that subjects 0-13 are used for train, 14-18 for validation and 
# 19-23 for testing
train_subjects = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0]

split_data, test_data = dfs.train_test_split(dataset = dataset,
                                              labels = ("id",), 
                                              subject_col=subject_col, 
                                              train_subjects=split_subjects,
                                              verbose=0)

# Split the remaining data into train and validation set, according to the trials.
# Trials 1-9 are used for training while 11-15 are used for validation, this ensures that
# each action is present in validation and training set
dfs2 = DataFrameSplitter(method="subject")
train_data, val_data = dfs2.train_test_split(dataset = split_data,
                                              labels = ("id",), 
                                              subject_col=subject_col, 
                                              train_subjects=train_subjects,
                                              verbose=0)

# print("[INFO] -- Segmenting data into windows")

"""
Creation of windowed data for MotionSense:
Segment all data into windows, of 128 samples per window (at 50 Hz 2.56 s).
Step size is also 128, which indicates no overlap between windows
The function get_windows takes care of segmenting the data into windows and
returns the raw data in dimesnionality num_windows x channels x win_size, 
the corresponding labels 

For the additional information the order is as follows:

subject_id: index of the subject ranging from 0-23, for subjects 1-24
weight: weight of the subject
height: height of the subject
age: age of the subject
gender: gender of the subject, 0 encodes female, 1 encodes male
trial_id: trial that this sample came from, note that this indicates the class as in each trial only one activity was
        performed, so this may need to be removed
        
the encoding of the class labels is as follows:
"dws": 0, "ups": 1, "wlk": 2, "jog": 3, "std": 4, "sit": 5
"""

def calc_logpriors(y_trn):
    N = len(y_trn)
    priors = np.array([len(np.where(y_trn == t)[0]) for t in np.unique(y_trn)])
    priors = priors/N
    return np.log(priors)

# Data Augmentation functions from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Scaling(X, sigma=0.1):
    np.random.seed(42)
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def DA_Jitter(X, sigma=0.05):
    np.random.seed(42)
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise
# End DA functions

def augment_data(X, y, p=0.5, jitter=0.05, scale=0.1):
    # augment training samples with noisy samples, here jittering and scaling
    augment_X = DA_Jitter(X, sigma=jitter)
    augment_X = DA_Scaling(augment_X, sigma=scale)
    
    idx = np.random.choice(augment_X.shape[0], int(augment_X.shape[0]*p), replace=False)
    augment_X = augment_X[idx]
    augment_y = y[idx]
    return augment_X, augment_y

def get_moSense(window=True, win_size=32, step_size=16, augment=True, noise=True):
    x_trn, y_trn, _ = get_windows(train_data, win_size, step_size)
    x_val, y_val, _ = get_windows(val_data, win_size, win_size)
    x_tst, y_tst, _ = get_windows(test_data, win_size, win_size)
    
    if noise:
        x_trn = DA_Jitter(x_trn, sigma=0.025)  # simulating sensor noise
    
    if augment:
        augment_X, augment_y = augment_data(x_trn, y_trn,p=1., jitter=0.05, scale=0.1)
        x_trn = np.concatenate((x_trn, augment_X), axis=0)
        y_trn = np.concatenate((y_trn, augment_y), axis=0)
    
    log_priors = calc_logpriors(y_trn)
    return x_trn, y_trn, x_val, y_val, log_priors, x_tst, y_tst, ACT_LABELS

x_trn, y_trn, x_val, y_val, log_priors, x_tst, y_tst, ACT_LABELS = get_moSense(window=True, win_size=1, step_size=1, augment=False)

print('MOS trn:',y_trn.shape)
print('MOS val:',y_val.shape)
print('MOS tst:',y_tst.shape)