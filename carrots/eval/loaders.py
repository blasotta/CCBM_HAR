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
from pathlib import Path
import glob


from sklearn.model_selection import train_test_split

CONFIG_PATH = '../config/'
BASEPATH = 'C:/Users/bened/PythonWork/CCBM_HAR/carrots'

pd.set_option('display.max_columns', None)

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def load_dataset(number):
    dataset, meta = arff.loadarff(
        BASEPATH + f'/data/001-IMU/raw_{number}.arff')
    labels = dataset['class']
    df = pd.DataFrame(dataset)
    df.drop(columns=['time', 'class'], inplace=True)

    for i in range(len(labels)):
        s = labels[i].decode('UTF-8')
        split = s.split('-', 1)
        labels[i] = split[0]

    str_labels = labels.astype('str')
    le = preprocessing.LabelEncoder()
    le.fit(str_labels)
    y = le.transform(str_labels)
    X = df.to_numpy()
    return X, y, le

# def load_power():
#     power = pd.read_csv('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/household_power_consumption.csv',
#                         sep=';', header=0, dtype={'Date': str, 'Time': str,
#                                                   'Global_active_power': np.float32,
#                                                   'Global_reactive_power': np.float32,
#                                                   'Voltage': np.float32,
#                                                   'Global_intensity': np.float32,
#                                                   'Sub_metering_1': np.float32,
#                                                   'Sub_metering_2': np.float32,
#                                                   'Sub_metering_3': np.float32})
#     power.drop(columns=['Date', 'Time'], inplace=True)
#     power.replace([np.inf, -np.inf], np.nan, inplace=True)
#     power.dropna(how="any", inplace=True)
    
#     x_trn, x_split = train_test_split(power,test_size=0.3,train_size=0.7)
#     x_tst, x_val = train_test_split(x_split,test_size=0.5,train_size=0.5)
    
#     return x_trn.to_numpy(), x_tst.to_numpy(), x_val.to_numpy()


def load_lara_trn():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/trn/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/trn/y/*.csv')
    

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    
    df = data_x.assign(Class=data_y)
    
    datasets = {} #dictionary with key=class and value=data pairs
    by_class = df.groupby('Class')

    for groups, data in by_class:
        d = data.drop(columns='Class')
        datasets[groups] = d

    return datasets, df.shape[0]

def load_lara_val():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/val/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/val/y/*.csv')
    

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    
    df = data_x.assign(Class=data_y)
    
    datasets = {} #dictionary with key=class and value=data pairs
    by_class = df.groupby('Class')

    for groups, data in by_class:
        d = data.drop(columns='Class')
        datasets[groups] = d

    return datasets, df.shape[0]

def load_lara_tst():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/tst/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/tst/y/*.csv')

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    
    X = data_x.to_numpy()
    y = data_y.to_numpy()
    classes = [0,1,2,3,4,5,6,7]
    return X, y, classes

def cond_lara_trn():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/trn/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/trn/y/*.csv')
    

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    data_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_x.dropna(how="any", inplace=True)
    
    # Normalization
    # Setting global variable so that the validation and test set can be
    # normalized with mean and std of train set as it should be done
    global mean
    global std
    
    df_norm = data_x.copy()
    
    mean = df_norm.mean(axis=0)
    std = df_norm.std(axis=0)

    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - mean[column]) / std[column]
        
    
    # corr_features = set()
    # corr_matrix = data_x.corr()
    
    # for i in range (len(corr_matrix.columns)):
    #     for j in range(i):
    #         if abs(corr_matrix.iloc[i,j]) > 0.8:
    #             colname = corr_matrix.columns[i]
    #             corr_features.add(colname)

    
    train_x = df_norm.to_numpy()
    train_y = data_y.to_numpy().flatten()
    return train_x, train_y

def cond_lara_val():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/val/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/val/y/*.csv')
    

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    data_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_x.dropna(how="any", inplace=True)
    
    df_norm = data_x.copy()
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - mean[column]) / std[column]
        
    #print(df_norm.describe())
    
    val_x = df_norm.to_numpy()
    val_y = data_y.to_numpy().flatten()
    return val_x, val_y

def cond_lara_tst():
    csv_trn_x = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/tst/x/*.csv')
    csv_trn_y = glob.glob('C:/Users/bened/PythonWork/CCBM_HAR/carrots/data/IMUdata_MbientLab/tst/y/*.csv')

    df_x = (pd.read_csv(file) for file in csv_trn_x)
    df_y = (pd.read_csv(file) for file in csv_trn_y)

    data_x = pd.concat(df_x, ignore_index=True)
    y_all = pd.concat(df_y, ignore_index=True)
    data_y = y_all[['Class']]
    
    data_x.drop(columns=['Time'], inplace=True)
    data_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_x.dropna(how="any", inplace=True)
    
    df_norm = data_x.copy()
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - mean[column]) / std[column]
    
    X = df_norm.to_numpy()
    y = data_y.to_numpy().flatten()
    classes = [0,1,2,3,4,5,6,7]
    return X, y, classes


def load_conditional_train():
    X_1, y_1, _ = load_dataset(1)
    X_2, y_2, _ = load_dataset(2)
    X_3, y_3, _ = load_dataset(3)
    X_5, y_5, _ = load_dataset(5)
    X_6, y_6, _ = load_dataset(6)

    train_x = np.concatenate([X_1, X_2, X_3, X_5, X_6], axis=0)
    train_y = np.concatenate([y_1, y_2, y_3, y_5, y_6], axis=0)
    
    global c_mean
    global c_std
    
    df = pd.DataFrame(train_x)
    df_norm = df.copy()
    
    c_mean = df_norm.mean(axis=0)
    c_std = df_norm.std(axis=0)
    
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - c_mean[column]) / c_std[column]
        
    train_x = df_norm.to_numpy()
    
    return train_x, train_y

def load_conditional_val():
    X_4, y_4, _ = load_dataset(4)

    val_x = X_4
    val_y = y_4
    
    df = pd.DataFrame(val_x)
    df_norm = df.copy()
    
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - c_mean[column]) / c_std[column]
        
    val_x = df_norm.to_numpy()

    return val_x, val_y

def get_carrots():
    X_1, y_1, _ = load_dataset(1)
    X_2, y_2, _ = load_dataset(2)
    X_3, y_3, _ = load_dataset(3)
    X_4, y_4, _ = load_dataset(4)
    X_5, y_5, _ = load_dataset(5)
    X_6, y_6, _ = load_dataset(6)
    #X_7, y_7, le = load_dataset(7)
    
    x = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis=0)
    y = np.concatenate([y_1, y_2, y_3, y_4, y_5, y_6], axis=0)
    
    x_trn, x_val, y_trn, y_val = train_test_split(x,y,test_size=0.2,
                                                      train_size=0.8,
                                                      random_state=42,
                                                      shuffle=True)
    # x_tst, x_val, y_tst, y_val = train_test_split(x_split, y_split,
    #                                               test_size=0.5,train_size=0.5,
    #                                               random_state=42,
    #                                               shuffle=True)
    
    priors = []
    N = y_trn.size
    for i in range(16):
        count = 0
        count = np.count_nonzero(y_trn == i)
        priors.append(count/N)
    
    log_priors = np.log(priors)
    
    global scaler
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(x_trn)
    valid_X = scaler.transform(x_val)
    #test_X = scaler.transform(x_tst)
    
    #add noise to train
    noise = np.random.normal(0, 0.33, size=(train_X.shape[0], train_X.shape[1]))
    train_X = train_X + noise
    #df = pd.DataFrame(train_X)
    # print(df.describe())
    
    return train_X, y_trn, valid_X, y_val, log_priors

# train_X, y_trn, valid_X, y_val, log_priors = get_carrots()

def load_conditional_test():
    X, y, le = load_dataset(7)
    
    tst_x = scaler.transform(X)

    return tst_x, y, le

def load_split_train():
    X_1, y_1, _ = load_dataset(1)
    X_2, y_2, _ = load_dataset(2)
    X_3, y_3, _ = load_dataset(3)
    X_5, y_5, _ = load_dataset(5)
    X_6, y_6, _ = load_dataset(6)

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

def load_split_val():
    X_4, y_4, _ = load_dataset(4)
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


def load_split_test():
    return load_dataset(7)

def one_hot_encode(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])