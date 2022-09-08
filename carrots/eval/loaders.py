# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:43:37 2022

@author: bened
"""

from scipy.io import arff
from sklearn import preprocessing
import pandas as pd
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

