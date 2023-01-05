import numpy as np
import pandas as pd
import itertools
from numpy.lib.stride_tricks import as_strided

pd.set_option('display.max_columns', None)

BASEPATH = 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval/'

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes
    """ 

    dss = pd.read_csv(BASEPATH+"data_subjects_info.csv")
    # print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    # print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = BASEPATH+'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset

def make_views(arr, win_size, step_size, writeable = False):
  """
  arr: any 2D array whose columns are distinct variables and 
    rows are data records at some timestamp t
  win_size: size of data window (given in data points along record/time axis)
  step_size: size of window step (given in data point along record/time axis)
  writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  
  Note that step_size is related to window overlap (overlap = win_size - step_size), in 
  case you think in overlaps.
  """
  
  n_records = arr.shape[0]
  n_columns = arr.shape[1]
  remainder = (n_records - win_size) % step_size 
  num_windows = 1 + int((n_records - win_size - remainder) / step_size)
  new_view_structure = as_strided(
    arr,
    shape = (num_windows, win_size, n_columns),
    strides = (8 * step_size * n_columns, 8 * n_columns, 8),
    writeable = False,
  )
  return new_view_structure

def get_windows(dataset, win_size, step_size):
    dat = []
    lab = []
    inf = []
    label_values = list()
    labels = ('id', 'trial')
    split_idx = dataset.shape[1] - 7
    # Get all cominations of subject ids and trials to calculate windows on
    for label in labels:
        unique_vals = sorted(dataset[label].unique())
        label_values.append(unique_vals)
    combinations = list(itertools.product(*label_values))
    
    for i, comb in enumerate(combinations):
        sub = dataset[(dataset['id'] == comb[0]) & (dataset['trial'] == comb[1])]
        data = sub.iloc[:, :split_idx]
        info = sub.iloc[0, split_idx:]
        arr = data.to_numpy()
        c = info.to_numpy()
        arr = np.ascontiguousarray(arr)
        c = np.ascontiguousarray(c)
        view_data = make_views(arr, win_size, step_size)
        subject_info = np.tile(c, (view_data.shape[0], 1))
        split = np.hsplit(subject_info, [1])
        label = split[0].flatten()
        sinfo = split[1]
        dat.append(view_data)
        lab.append(label)
        inf.append(sinfo)
    
    data = np.concatenate(dat)
    data = data.reshape(data.shape[0], -1)
    labels = np.concatenate(lab)
    info = np.concatenate(inf)
    
    return data, labels, info
