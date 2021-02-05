import numpy as np 
import pandas as pd

from itertools import combinations
from itertools import product
from collections import deque
from sklearn.model_selection import LeaveOneOut

#Local imports
from tfg.naive_bayes import NaiveBayes


def memoize(f):
    cache =dict()
    def g(clf,cv,X,y,columns):
        elements = frozenset(columns)
        if elements not in cache:
            cache[elements] = f(clf,cv,X,y,columns)
        return cache[elements]
    return g

# @memoize
def _evaluate(clf,cv,X,y,columns):
    if isinstance(X,pd.DataFrame):
        X=X.to_numpy()
    scores = []
    if cv:
        for train_index, test_index in cv.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train,y_train)
            score = clf.score(X_test,y_test)
            scores.append(score)
        return np.mean(scores)
    return clf.leave_one_out_cross_val(X,y)
    
concat = lambda d: "-".join(d)

def _join_columns(X,columns):
    if isinstance(X,pd.DataFrame):
        X=X.to_numpy()
    X_1 = None
    X_2 = X.astype(str)
    for col in columns:
        if isinstance(col,tuple):
            idx = list(col)
            if X_1 is not None:
                X_1= np.concatenate([X_1,np.apply_along_axis(concat, 1, X_2[:,idx]).reshape(-1,1)],axis=1)
            else:
                X_1 = np.apply_along_axis(concat, 1, X_2[:,idx]).reshape(-1,1)
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

class PazzaniWrapper:
    def __init__(self,seed=None,cv = None, strategy = "BSEJ",verbose=0):
        self.cv = cv
        self.classifier = NaiveBayes(encode_data=True)
        self.seed = seed
        self.strategy = strategy
        allowed_strategies = ("BSEJ","FSSJ")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown strategy type: %s, expected one of %s." % (self.strategy, allowed_strategies))
        self.search = self.search_bsej if strategy=="BSEJ" else self.search_fssj
        self.verbose = verbose
    def combine_columns(self,X,columns=None):
        if columns:
            return np.apply_along_axis(concat, 1, X[:,columns]).reshape(-1,1)
        return np.apply_along_axis(concat, 1, X).reshape(-1,1)

    def _generate_neighbors_bsej(self,current_columns,X):
        if X.shape[1]>1:
            for col in range(X.shape[1]):
                new_columns = current_columns.copy()
                del new_columns[col]
                yield new_columns,np.delete(X,col,axis=1)
            for features in combinations(np.arange(X.shape[1]),2):
                new_col_name = flatten([current_columns[features[0]],current_columns[features[1]]])
                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                features = sorted(features,reverse=True)
                del new_columns[features[0]]
                del new_columns[features[1]]
                
                columns = features
                combined_columns = self.combine_columns(X,columns)
                neighbor = np.concatenate([X,combined_columns],axis=1)
                yield new_columns, np.delete(neighbor,columns,axis=1)
     
    def search_bsej(self,X,y):
        self.evaluate = memoize(_evaluate)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.astype(str)
        current_best = X.copy()
        current_columns = deque(range(X.shape[1]))
        best_score=self.evaluate(self.classifier,self.cv,current_best,y,current_columns)
        stop=False
        while not stop:
            stop=True
            if self.verbose:
                print("Current Best: ", current_columns, " Score: ",best_score)
            for new_columns,neighbor in self._generate_neighbors_bsej(current_columns,current_best):
                score=self.evaluate(self.classifier,self.cv,neighbor,y,new_columns)
                if self.verbose==2:
                    print("\tNeighbor: ", new_columns, " Score: ",score)
                if score > best_score:
                    stop=False
                    current_best = neighbor
                    best_score = score
                    current_columns = new_columns
                    if score == 1.0:
                        stop=True
                        break

        print("Final best: ", list(current_columns), " Score: ",best_score)
        model = self.classifier.fit(current_best,y)
        features = current_columns
        transformer = lambda X: _join_columns(X,columns = features)
        return transformer, features, model

    def _generate_neighbors_fssj(self,current_columns, individual , original_data, available_columns):
        if available_columns:
            for index,col in enumerate(available_columns):
                new_columns = current_columns.copy()
                new_columns.append(col)
                new_available_columns = available_columns.copy()
                del new_available_columns[index]
                if individual is not None:
                    neighbor =  np.concatenate([individual,original_data[:,col].reshape(-1,1)],axis=1)
                else:
                    neighbor = original_data[:,col].reshape(-1,1)
                yield new_columns,new_available_columns,neighbor
        if individual is not None and individual.shape[1]>0 and available_columns:
            for features_index in product(np.arange(len(available_columns)),np.arange(len(current_columns))):
                features  = available_columns[features_index[0]],current_columns[features_index[1]]
                new_col_name = flatten([features[0],features[1]])
                
                new_available_columns = available_columns.copy()
                del new_available_columns[features_index[0]]

                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                del new_columns[features_index[1]]

                if isinstance(features[1],tuple):
                    features[1] = list(features[1])
                separated_columns = np.concatenate([original_data[:,features[0]].reshape(-1,1),individual[:,features_index[1]].reshape(-1,1)],axis=1)
                combined_columns = self.combine_columns(separated_columns)
                neighbor = np.concatenate([individual,combined_columns],axis=1)
                yield new_columns,new_available_columns, np.delete(neighbor,features_index[1],axis=1)

    def search_fssj(self,X,y):
        self.evaluate = memoize(_evaluate)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.astype(str)
        current_best = None
        current_columns = deque()
        available_columns = list(range(X.shape[1]))
        best_score= -float("inf")
        stop=False
        while not stop:
            stop=True
            if self.verbose:
                print("Current Best: ", current_columns, " Score: ",best_score,"Available columns: ", available_columns)
            for new_columns,new_available_columns,neighbor in self._generate_neighbors_fssj(current_columns = current_columns,
                                                                                            individual = current_best,
                                                                                            original_data = X,
                                                                                            available_columns = available_columns):
                score = self.evaluate(self.classifier,self.cv,neighbor,y,new_columns)  
                if self.verbose==2:
                    print("\tNeighbour: ", new_columns, " Score: ",score,"Available columns: ", new_available_columns)
                if score > best_score:
                    stop=False
                    current_best = neighbor
                    best_score = score
                    current_columns = new_columns
                    available_columns = new_available_columns
                    if score == 1.0:
                        stop=True
                        break
        print("Final best: ", list(current_columns), " Score: ",best_score)
        model = self.classifier.fit(current_best,y)
        features = current_columns
        transformer = lambda X: _join_columns(X,columns = features)
        return transformer, features, model

    def evaluate(self,classifier,cv,X,y,columns):
        return _evaluate(classifier,cv,X,y,columns)