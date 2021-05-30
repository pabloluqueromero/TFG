import pandas as pd
import numpy as np
import os
import plotly.express as px

from collections import OrderedDict, deque
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
from json import dumps

from tfg.feature_construction import DummyFeatureConstructor

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



def shannon_entropy(column):
    count = np.bincount(column)
    return entropy(count, base=2)


def info_gain(X,y,feature=0):
    '''Info gain: H(Y) - H(X|Y)'''
    H_C = shannon_entropy(y)
    H_X_C = 0
    feature_values = np.unique(X[:,feature])
    for value in feature_values:
        mask = X[:,feature]==value
        prob = mask.sum()/X.shape[0]
        H_X_C += prob*shannon_entropy(y[mask])
    return H_C - H_X_C


def symmetrical_uncertainty(f1,f2,X=None):
    '''SU 
       Formula: 2*I(X,Y)/(H(X)+H(Y)
       f1: feature 1 (int or array-like),
       f2: feature 2, (int or array-like)''' 
    if (isinstance(f1,int) or isinstance(f2,int)) and X is None:
        raise Exception("X must be provided for feature indexation")
    a = X[:,f1] if isinstance(f1,int) else f1
    b = X[:,f2] if isinstance(f2,int) else f2
    # return normalized_mutual_info_score(a.flatten(),b.flatten())
    a = a.flatten()
    b = b.flatten()
    gain = info_gain(a.reshape(-1,1),b)
    H_a = shannon_entropy(a)
    H_b = shannon_entropy(b)
    if (H_a+H_b)==0:
        return 0
    return 2*(gain)/(H_a+H_b)



def compute_sufs(current_su,current_features,new_feature,y,beta=0.5,minimum=None):
    '''
    MIFS adapted to work with SU.
    Example:
        SU({X1,X2,X3}|Y) = sum(SU(Xi|Y)) - beta * (SU(X1,X2),SU(X2,X3))
    '''
    class_su = symmetrical_uncertainty(f1=new_feature,f2=y)
    if current_features is None:
        penalisation = 0
    else:
        penalisation = beta*sum(
                    symmetrical_uncertainty(current_features[:,j],new_feature) #->The result should be the same but sklearn's is more tested
                    # normalized_mutual_info_score(current_features[:,j].flatten(),new_feature.flatten())
                    for j in range(current_features.shape[1]))

    su = current_su+class_su-penalisation 
    return su if minimum is None else max(su,minimum)


def compute_sufs_non_incremental(features,y,beta=0.5,minimum=None):
    '''
    MIFS adapted to work with SU.
    Example:
        SU({X1,X2,X3}|Y) = sum(SU(Xi|Y)) - beta * (SU(X1,X2),SU(X2,X3))
    '''
    class_su = sum([symmetrical_uncertainty(f1=f,f2=y) for f in features])
    penalisation = beta*sum(
                    symmetrical_uncertainty(f1,f2) #->The result should be the same but sklearn's is more tested
                    # normalized_mutual_info_score(current_features[j],new_feature)
                    for f1,f2 in combinations(features,2))

    su = class_su-penalisation 
    return su if minimum is None else max(su,minimum)




def translate_features(features,feature_encoder,categories=None,path=".",filename="selected_features"):
    if path[-1]=="/":
        path = path[:-1]

    translated_features = []
    with open(f"{path}/{filename}.json", 'w') as f:
        for feature in features:
            od = OrderedDict()
            od["type"] = "DummyFeature" if isinstance(feature,DummyFeatureConstructor)  else "LogicalFeature" 
            od["detail"] = feature.get_dict_translation(feature_encoder,categories)
            translated_features.append(od)
        f.write(dumps(translated_features)) 


# def mutual_information_class_conditioned(f1,f2,y):
#     values, counts = np.unique(y,return_counts=True)
#     counts = counts/counts.sum()
#     score = []
#     for i in range(values.shape[0]):
#         value = values[i]
#         mask = y == value
#         score.append(normalized_mutual_info_score(f1[mask],f2[mask]))
#     score = np.array(score)
#     return (score * counts).sum()


def mutual_information_class_conditioned2(f1,f2,y):
    values, counts = np.unique(y,return_counts=True)
    counts = counts/counts.sum()
    score = []
    for i in range(values.shape[0]):
        value = values[i]
        mask = y == value
        score.append(normalized_mutual_info_score(f1[mask],f2[mask]))
    score = np.array(score)
    return (score * counts).sum()
    
def mutual_information_class_conditioned(f1,f2,y):
    X = combine_columns(np.concatenate([f1.reshape(-1,1),f2.reshape(-1,1)],axis=1).astype(str)).flatten()
    # values, counts = np.unique(y,return_counts=True)
    # counts = counts/counts.sum()
    # score = []
    # for i in range(values.shape[0]):
    #     value = values[i]
    #     mask = y == value
    #     score.append(normalized_mutual_info_score(f1[mask],f2[mask]))
    # score = np.array(score)
    # return (score * counts).sum()
    return normalized_mutual_info_score(X,y)



def get_X_y_from_database(base_path, name, data, test, label):
    full_data_path = base_path+name+"/"+data
    full_test_path = base_path+name+"/"+test
    has_test = os.path.exists(base_path+name+"/"+test)
    assert pd.read_csv(full_data_path)[label].name == label
    if has_test:
        train = pd.read_csv(full_data_path)
        test = pd.read_csv(full_test_path)
        df = train.append(test)

    else:
        df = pd.read_csv(full_data_path)
    X = df.drop([label], axis=1)
    y = df[label]
    return X, y



def get_graphs(df,folder):
    #FIT
    filename = "fit_time_fix_n_samples.png"
    fig = px.line(df[df["n_samples"].isin([10,100,1000])].sort_values(['n_samples','n_features']),
                x="n_features", 
                y="Average Fit Time", 
                color='Classifier',
                facet_col="n_samples", 
                width=1000,)
    fig.write_image(folder+filename)

    filename = "fit_time_fix_n_features.png"
    fig = px.line(df[df["n_features"].isin([10,100,1000])].sort_values(['n_features','n_samples']),
                x="n_samples", 
                y="Average Fit Time", 
                color='Classifier',
                facet_col="n_features", 
                width=1000)
    fig.write_image(folder+filename)
    
    #Predict
    filename = "predict_time_fix_n_samples.png"
    fig = px.line(df[df["n_samples"].isin([10,100,1000])].sort_values(['n_samples','n_features']),
                x="n_features", 
                y="Average Predict Time", 
                color='Classifier',
                facet_col="n_samples", 
                width=1000,)
    fig.write_image(folder+filename)

    filename = "predict_time_fix_n_features.png"
    fig = px.line(df[df["n_features"].isin([10,100,1000])].sort_values(['n_features','n_samples']),
                x="n_samples", 
                y="Average Predict Time", 
                color='Classifier',
                facet_col="n_features", 
                width=1000,)
    fig.write_image(folder+filename)
    
def get_scorer(scoring):
    scores = {"accuracy": accuracy_score,
              "f1_score": f1_score,
              "roc_auc_score":roc_auc_score
    }
    if scoring in scores:
        return scores[scoring]
    raise ValueError(f"The specified scoring {scoring} is not valid. Expected one of {tuple(scores.keys())}")


def transform_features(features,X):
    # return np.concatenate([f.transform(X) for f in features],axis=1)
    array = np.zeros(shape=(X.shape[0],len(features)))
    for i,f in enumerate(features):
        transformed = f.transform(X).reshape(-1,1)
        array[:,i] = transformed[:,0]
    return array



def backward_search(X,y,current_features,classifier):
    check_is_fitted(classifier)
    transformed_features = np.concatenate([f.transform(X) for f in current_features],axis=1)
    improvement = True
    best_score = classifier.leave_one_out_cross_val(transformed_features,y,fit=False)
    while improvement and transformed_features.shape[1] >1:
        improvement = False
        feature = None
        for i in range(transformed_features.shape[1]):
            feature = transformed_features[:,i].reshape(-1,1)
            iteration_features = np.delete(transformed_features,i,axis=1)
            classifier.remove_feature(i)
            current_score = classifier.leave_one_out_cross_val(iteration_features,y,fit=False)
            classifier.add_features(feature,y,[i])
            if current_score > best_score:
                feature_index = i
                improvement = True
                best_score = current_score

            
        if improvement:
            transformed_features = np.delete(transformed_features,feature_index,axis=1)
            classifier.remove_feature(feature_index)
            del current_features[feature_index]
    return current_features


def hash_features(features):
    return hash(tuple(features))

def append_column_to_numpy(array,column):
    a = np.zeros((array.shape[0],array.shape[1]+1),dtype=int)
    a[:,:-1] = array
    a[:,-1]=column.flatten()
    return a