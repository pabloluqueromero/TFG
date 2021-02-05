import numpy as np 
import pandas as pd

from itertools import combinations
from itertools import product
from collections import deque
from sklearn.model_selection import LeaveOneOut

#Local imports
from tfg.naive_bayes import NaiveBayes
from tfg.wrapper import PazzaniWrapper

def memoize(f):
    cache =dict()
    def g(clf,X,y,columns,fit):
        elements = frozenset(columns)
        if elements not in cache:
            cache[elements] = f(clf,X,y,columns,fit)
        return cache[elements]
    return g

# @memoize
def _evaluate(clf,X,y,columns,fit):
    if isinstance(X,pd.DataFrame):
        X=X.to_numpy()
    return clf.leave_one_out_cross_val(X,y,fit=fit)
    
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

class PazzaniWrapperNB(PazzaniWrapper):
    def __init__(self,seed=None, strategy = "BSEJ",verbose=0):
        super().__init__(seed=seed, strategy=strategy, verbose=verbose,cv=None)

    def _generate_neighbors_bsej(self,current_columns,X):
        if X.shape[1]>1:
            for column_to_drop in range(X.shape[1]):
                new_columns = current_columns.copy()
                del new_columns[column_to_drop]
                yield new_columns,column_to_drop,None,True #Updated column list, columns to remove, columns to add, delete?
            for features in combinations(np.arange(X.shape[1]),2):
                new_col_name = flatten([current_columns[features[0]],current_columns[features[1]]])
                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                columns_to_drop = sorted(features,reverse=True)
                del new_columns[columns_to_drop[0]]
                del new_columns[columns_to_drop[1]]
                
                combined_columns = self.combine_columns(X,list(features))
                yield new_columns,list(columns_to_drop), combined_columns, False
     
    def search_bsej(self,X,y):
        self.evaluate = memoize(_evaluate)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.astype(str)
        current_best = X.copy()
        current_columns = deque(range(X.shape[1]))
        best_score=self.evaluate(self.classifier,current_best,y,current_columns,fit=True)
        stop=False
        update=False
        while not stop:
            stop=True
            if self.verbose:
                print("Current Best: ", current_columns, " Score: ",best_score)
            for new_columns,columns_to_delete,columns_to_add,delete in self._generate_neighbors_bsej(current_columns,current_best):
                if delete:
                    action = "DELETE"
                    #Update classifier and get validation result
                    self.classifier.remove_feature(columns_to_delete)
                    neighbor = np.delete(current_best,columns_to_delete,axis=1)
                    score=self.evaluate(self.classifier,neighbor,y,new_columns,fit=False)

                    #Restore the column for the next iteration
                    self.classifier.add_features(current_best[:,columns_to_delete].reshape(-1,1),y,index=[columns_to_delete])
                else:
                    action = "ADD"
                    self.classifier.remove_feature(columns_to_delete[0])
                    self.classifier.remove_feature(columns_to_delete[1])
                    
                    self.classifier.add_features(columns_to_add,y)

                    neighbor = np.delete(current_best,columns_to_delete,axis=1)
                    neighbor = np.concatenate([neighbor,columns_to_add],axis=1)

                    score=self.evaluate(self.classifier,neighbor,y,new_columns,fit=False)

                    self.classifier.remove_feature(neighbor.shape[1]-1)
                    self.classifier.add_features(current_best[:,columns_to_delete],y,index=columns_to_delete) #We reverse it for insert order
                
                if self.verbose==2:
                    print("\tNeighbor: ", new_columns, " Score: ",score)
                if score > best_score:
                    stop=False
                    best_columns = new_columns
                    best_action = action
                    best_score = score
                    best_columns_to_delete = columns_to_delete
                    update=True
                    if best_action == "ADD":
                        best_columns_to_add = columns_to_add
                    if score == 1.0:
                        stop=True
                        break
            if update:
                current_columns = best_columns
                if best_action == "DELETE":
                    current_best = np.delete(current_best,best_columns_to_delete,axis=1)
                    #Update best
                    self.classifier.remove_feature(best_columns_to_delete)
                else:
                    current_best = np.delete(current_best,best_columns_to_delete,axis=1)
                    current_best = np.concatenate([current_best,best_columns_to_add],axis=1)
                    #Update classifier
                    self.classifier.remove_feature(best_columns_to_delete[0])
                    self.classifier.remove_feature(best_columns_to_delete[1])
                    self.classifier.add_features(best_columns_to_add,y)

        print("Final best: ", list(current_columns), " Score: ",best_score)
        model = self.classifier
        features = current_columns
        transformer = lambda X: _join_columns(X,columns = features)
        return transformer, features, model

    #TODO
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
                separated_columns = np.concatenate([original_data[:,features_index[0]].reshape(-1,1),individual[:,features_index[1]].reshape(-1,1)],axis=1)
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

    def evaluate(self,classifier,X,y,columns,fit):
        return _evaluate(classifier,X,y,columns,fit)