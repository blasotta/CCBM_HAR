# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:43:37 2022

@author: bened
"""

from scipy.io import arff
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import yaml

CONFIG_PATH = '../config/'
BASEPATH = 'C:/Users/bened/PythonWork/CCBM_HAR/carrots'

def load_dataset(number):
    dataset, meta = arff.loadarff(BASEPATH + f'/data/001-IMU/raw_{number}.arff')
    labels = dataset['class']
    df = pd.DataFrame(dataset)
    df.drop(columns=['time', 'class'], inplace=True)
    
    for i in range(len(labels)):
        s = labels[i].decode('UTF-8')
        split = s.split('-',1)
        labels[i]=split[0]
    
    str_labels = labels.astype('str')
    le = preprocessing.LabelEncoder()
    le.fit(str_labels)
    y = le.transform(str_labels)
    X = df.to_numpy()
    return X,y,le

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def load_conditional_train():
    X_1, y_1,_ = load_dataset(1)
    X_2, y_2,_ = load_dataset(2)
    X_3, y_3,_ = load_dataset(3)
    X_5, y_5,_ = load_dataset(5)
    X_6, y_6,_ = load_dataset(6)
    
    train_x = np.concatenate([X_1, X_2, X_3, X_5, X_6], axis=0)
    train_y = np.concatenate([y_1, y_2, y_3, y_5, y_6], axis=0)
    
    D = np.c_[train_x, train_y]
    length = np.shape(D)[0]
    
    df = pd.DataFrame(D)
    df.rename(columns={30: "class"}, inplace = True)
    data = df.astype({'class': 'int32'})
    
    #group by class for conditional density estimation.
    datasets = {} #dictionary with key=class and value=data pairs
    by_class = df.groupby('class')
    
    for groups, data in by_class:
        d = data.drop(columns='class')
        datasets[groups] = d
    
    
    return datasets, length

def load_conditional_val():
    X_4, y_4,_ = load_dataset(4)
    
    val_x = X_4
    val_y = y_4
    
    D = np.c_[val_x, val_y]
    length = np.shape(D)[0]
    
    df = pd.DataFrame(D)
    df.rename(columns={30: "class"}, inplace = True)
    data = df.astype({'class': 'int32'})
    
    #group by class for conditional density estimation.
    datasets = {} #dictionary with key=class and value=data pairs
    by_class = df.groupby('class')
    
    for groups, data in by_class:
        d = data.drop(columns='class')
        datasets[groups] = d
    
    
    return datasets, length

def load_conditional_test():
    return load_dataset(7)