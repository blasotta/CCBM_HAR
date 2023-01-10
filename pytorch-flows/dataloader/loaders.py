from scipy.io import arff
from scipy import stats
from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided


from sklearn.model_selection import train_test_split

BASEPATH = 'C:/Users/bened/PythonWork/Test_MA/pytorch-flows/dataloader'

def load_dataset(number):
    dataset, meta = arff.loadarff(
        BASEPATH + f'/data/Carrots/raw_{number}.arff')
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

def one_hot_encode(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

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


def make_views(arr, win_size, step_size, writeable = False):
  """
  arr: input 2d array to be windowed
  win_size: size of data window (given in data points)
  step_size: size of window step (given in data point)
  writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  Note that data should be of type 64 bit (8 byte)
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

def get_windows(X,y, win_size, step_size):
    X = np.ascontiguousarray(X)
    views = make_views(X, win_size, step_size)
    x_newshape = (views.shape[0], views.shape[1]*views.shape[2])
    x = np.reshape(views, newshape=x_newshape)
        
    y = y.astype('int64') # change labels to int64 for windowing correctly
    y = np.ascontiguousarray(y)
    y_2d = np.reshape(y, (y.shape[0],1))
    y_views = make_views(y_2d, win_size, step_size)
    y_newshape = (y_views.shape[0], y_views.shape[1]*y_views.shape[2])
    y_new = np.reshape(y_views, newshape=y_newshape)
    y,_ = stats.mode(y_new, axis=1, keepdims=False)
    
    assert (y.shape[0] == x.shape[0])
    
    return x,y

def augment_data(X, y, p=0.5, jitter=0.05, scale=0.1):
    # augment training samples with noisy samples, here jittering and scaling
    augment_X = DA_Jitter(X, sigma=jitter)
    augment_X = DA_Scaling(augment_X, sigma=scale)
    
    idx = np.random.choice(augment_X.shape[0], int(augment_X.shape[0]*p), replace=False)
    augment_X = augment_X[idx]
    augment_y = y[idx]
    return augment_X, augment_y

def calc_logpriors(y_trn):
    N = len(y_trn)
    priors = np.array([len(np.where(y_trn == t)[0]) for t in np.unique(y_trn)])
    priors = priors/N
    return np.log(priors)


# Load carrots Dataset, it has 16 distinct classes 0-15
def get_carrots(window=True, win_size=32, step_size=16, augment=True, noise=True):
    X_1, y_1, _ = load_dataset(1)
    X_2, y_2, _ = load_dataset(2)
    X_3, y_3, _ = load_dataset(3)
    X_4, y_4, _ = load_dataset(4)
    X_5, y_5, _ = load_dataset(5)
    X_6, y_6, _ = load_dataset(6)
    
    if window:
        X_1,y_1 = get_windows(X_1, y_1, win_size, step_size)
        X_2,y_2 = get_windows(X_2, y_2, win_size, step_size)
        X_3,y_3 = get_windows(X_3, y_3, win_size, step_size)
        X_4,y_4 = get_windows(X_4, y_4, win_size, step_size)
        X_5,y_5 = get_windows(X_5, y_5, win_size, step_size)
        X_6,y_6 = get_windows(X_6, y_6, win_size, step_size)
    
    x = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis=0)
    y = np.concatenate([y_1, y_2, y_3, y_4, y_5, y_6], axis=0)
    
    global scaler
    scaler = preprocessing.MinMaxScaler() # scales variables in Interval [0.1]
    x = scaler.fit_transform(x)
    
    
    x_trn, x_val, y_trn, y_val = train_test_split(x,y,test_size=0.2,
                                                      train_size=0.8,
                                                      random_state=42,
                                                      shuffle=True)
    
    if noise:
        x_trn = DA_Jitter(x_trn, sigma=0.025)  # simulating sensor noise
    
    if augment:
        augment_X, augment_y = augment_data(x_trn, y_trn,p=1., jitter=0.05, scale=0.1)
        x_trn = np.concatenate((x_trn, augment_X), axis=0)
        y_trn = np.concatenate((y_trn, augment_y), axis=0)
    
    log_priors = calc_logpriors(y_trn)
    
    
    return x_trn, y_trn, x_val, y_val, log_priors


def load_conditional_test(window=True, win_size=32, step_size=32):
    tst_x, tst_y, le = load_dataset(7)
    
    if window:
        tst_x,tst_y = get_windows(tst_x, tst_y, win_size, step_size)
    
    tst_x = scaler.transform(tst_x)

    return tst_x, tst_y, le


# Load the UCI HAR Dataset, it has 6 distinct classes 1-6
def get_UCIHAR():
    classes = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']
    train_x = np.loadtxt(BASEPATH+'/data/UCI HAR/train/X_train.txt')
    train_y = np.loadtxt(BASEPATH+'/data/UCI HAR/train/y_train.txt')
    # reindex labels such that they start at 0
    train_y = train_y - np.min(train_y)
    
    x_trn, x_val, y_trn, y_val = train_test_split(train_x, train_y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  shuffle=True)

    log_priors = calc_logpriors(y_trn)
    
    x_tst = np.loadtxt(BASEPATH+'/data/UCI HAR/test/X_test.txt')
    y_tst = np.loadtxt(BASEPATH+'/data/UCI HAR/test/y_test.txt')
    # reindex labels such that they start at 0
    y_tst -= np.min(y_tst)
    
    return x_trn, y_trn, x_val, y_val, log_priors, x_tst, y_tst, classes