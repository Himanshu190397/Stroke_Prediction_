import numpy as np
import pandas as pd
import math

NUMERIC_STABILITY = 10**(-7)
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def init_weights(nIn,rng, bias= True, high=None, low=None):
    
    if high==None:
        high=(10**(-4))
    if low ==None:
        low= -high   
    
    omega = high*rng.random((nIn,1)) - high/2
    if bias:
        b = 0
        return omega, b
    else:
        return omega

def zscore(data, meanX=None, stdX=None, ddof=0, metrics=False):
    if (meanX is None ) or (stdX is None):
        meanX = np.mean(data, axis=0)            
        stdX = np.std(data, axis=0,ddof=ddof)
        stdX[stdX == 0] = 1
        
    tdata = np.zeros((data.shape))
    tdata = (data - meanX) / stdX
    tdata[tdata == 0] = 1

    if metrics ==True:
        return tdata, meanX, stdX
    else:
        return tdata

def load_dataset(path,rng, shuffle = True):
    data = np.genfromtxt(path, delimiter=',')
    # temp = [1]
    # t = temp*len(data)
    # data = np.insert(data,0,t,axis=1)
    if shuffle:
        rng.shuffle(data)
    return data    

def get_train_val_split(data, labels=None, return_split_index=False):
    s_idx = int(math.ceil(len(data)* (2/3)))
    if isinstance(data, pd.DataFrame):
        train_ds = data.iloc[:s_idx,:]
        val_ds = data.iloc[s_idx:,:]        
    else:
        train_ds = data[:s_idx,:]
        val_ds = data[s_idx:,:]    
    
    if return_split_index:
        return train_ds, val_ds, s_idx
    else:
        return train_ds, val_ds

def pandas_get_train_val(data, labels=None):
    s_idx = int(math.ceil(len(data)* (2/3)))
    train_ds = data.iloc[:s_idx,:]
    val_ds = data.iloc[s_idx:,:]    
        
    return train_ds, val_ds, s_idx

def split_x_y(dataset, yidx):
    if yidx == -1:
        yidx = dataset.shape[1] -1
    all_except = list(range(dataset.shape[1]))
    all_except.pop(yidx)
    if isinstance(dataset,pd.DataFrame):
        x = dataset.iloc[:,all_except]
        y = dataset.iloc[:,yidx].reshape(-1,1)
    else:
        x = dataset[:,all_except]
        y = dataset[:,yidx].reshape(-1,1)

    return x, y

def stdev(numbers):
    # sample standard deviation
    avg = np.mean(numbers,axis=0)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return np.sqrt(variance)

def covariance(data):
    cov = (data.T@data)/(len(data)-1)
    return cov

def onehotencode(label_vec):
    encoded_arr = np.zeros((label_vec.size, label_vec.max()+1))
    encoded_arr[np.arange(label_vec.size),label_vec]=1
    return encoded_arr

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])
