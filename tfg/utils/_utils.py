import pandas as pd
import numpy as np
from collections import deque

def concat_columns(d):
    return "-".join(d)

def join_columns(X,columns):
    if isinstance(X,pd.DataFrame):
        X=X.to_numpy()
    X_1 = None
    X_2 = X.astype(str)
    for col in columns:
        if isinstance(col,tuple):
            idx = list(col)
            if X_1 is not None:
                X_1= np.concatenate([X_1,np.apply_along_axis(concat_columns, 1, X_2[:,idx]).reshape(-1,1)],axis=1)
            else:
                X_1 = np.apply_along_axis(concat_columns, 1, X_2[:,idx]).reshape(-1,1)
        else:
            if X_1 is not None:
                X_1 = np.concatenate([X_1,X_2[:,col].reshape(-1,1)],axis=1)
            else:
                X_1 = X_2[:,col].reshape(-1,1)
    return X_1

def flatten(l):
    if l:
        q = flatten(l[1:])
        if hasattr(l[0],"__iter__"):
            return flatten(l[0]) + q
        q.appendleft(l[0])
        return q
    return deque()

def memoize(f,attribute_to_cache):
    cache =dict()
    def g(*args, **kwargs):
        elements = frozenset(kwargs[attribute_to_cache])
        if elements not in cache:
            cache[elements] = f(*args, **kwargs)
        return cache[elements]
    return g

def combine_columns(X,columns=None):
    if columns:
        return np.apply_along_axis(concat_columns, 1, X[:,columns]).reshape(-1,1)
    return np.apply_along_axis(concat_columns, 1, X).reshape(-1,1)

def make_discrete(X,m=100):
    X*=m
    minimum = np.amin(X)
    if minimum <0:
        minimum*=-1
        X+= minimum
    return X.astype(int)

def generate_xor_data():
    data=np.array(
        ([[1,1,'+']]*10)+
        ([[1,0,'-']]*10)+
        ([[0,1,'-']]*10)+
        ([[0,0,'+']]*10)
    )
    return data[:,:2],data[:,2]

def twospirals(n_points, noise=.5):
    """
    Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))).astype(int)) 


def onecold(a):
    n = len(a)
    s = a.strides[0]
    strided = np.lib.stride_tricks.as_strided
    b = np.concatenate((a,a[:-1]))
    return strided(b[1:], shape=(n-1,n), strides=(s,s))
    
def combinations_without_repeat(a):
    n = len(a)
    out = np.empty((n,n-1,2),dtype=a.dtype)
    out[:,:,0] = np.broadcast_to(a[:,None], (n, n-1))
    out.shape = (n-1,n,2)
    out[:,:,1] = onecold(a)
    out.shape = (-1,2)
    return out  
