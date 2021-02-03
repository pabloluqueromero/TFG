import numpy as np 
import pandas as pd

from itertools import combinations
from collections import deque
from sklearn.model_selection import LeaveOneOut,StratifiedKFold

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
                X_1= np.concatenate([X_1,np.apply_along_axis(concat, 1, X_2[:,col].reshape(-1,1)).reshape(-1,1)],axis=1)
            else:
                X_1 = np.apply_along_axis(concat, 1, X_2[:,col].reshape(-1,1)).reshape(-1,1)
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
    def __init__(self,seed=None,cv = None):
        self.cv = cv
        self.classifier = NaiveBayes(encode_data=True)
        self.seed = seed
        
    def combine_columns(self,X,columns):
        return np.apply_along_axis(concat, 1, X[:,columns]).reshape(-1,1)

    def _generate_neighbors(self,current_columns,X):
        current_col_set = set(current_columns)
        if X.shape[1]>1:
            for col in range(X.shape[1]):
                new_columns = current_columns.copy()
                del new_columns[col]
                np.delete(X,0,axis=1)
                yield new_columns,np.delete(X,col,axis=1)
            for features in combinations(np.arange(X.shape[1]),2):
                new_col_name = flatten([current_columns[features[0]],current_columns[features[1]]])
                if frozenset(new_col_name) in current_col_set:
                    continue
                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                del new_columns[features[0]]
                del new_columns[features[1]-(1 if features[0] < features[1] else 0)]
                
                columns = list(features)
                combined_columns = self.combine_columns(X,columns)
                neighbor = np.concatenate([X,combined_columns],axis=1)
                yield new_columns, np.delete(neighbor,columns,axis=1)
            
    def search(self,X,y):
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
            print("Current_best: ", current_columns, " Score: ",best_score)
            for columns,neighbor in self._generate_neighbors(current_columns,current_best):
                score=self.evaluate(self.classifier,self.cv,neighbor,y,columns)
                if score > best_score:
                    stop=False
                    current_best = neighbor
                    best_score = score
                    current_columns = columns
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

if __name__ == "__main__":
    seed = 200
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification, make_blobs, make_moons,make_circles,make_checkerboard,make_swiss_roll
    def make_discrete(X,m=100):
        X*=m
        minimum = np.amin(X)
        if minimum <0:
            minimum*=-1
            X+= minimum
        return X.astype(int)
    # X,y = make_classification(n_samples=100, 
    #                       n_features=10, 
    #                       n_informative=7, 
    #                       n_redundant=0, 
    #                       n_repeated=0, 
    #                       n_classes=2, 
    #                       n_clusters_per_class=2, 
    #                       weights=None,
    #                       class_sep=1.0, 
    #                       hypercube=True, 
    #                       scale=2.0, 
    #                       shuffle=True, 
    #                       random_state=seed)
    # X = make_discrete(X,m=10)
    np.random.seed(200)
    def twospirals(n_points, noise=.5):
        """
        Returns the two spirals dataset.
        """
        n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
        d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
        d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
        return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
                np.hstack((np.zeros(n_points),np.ones(n_points))).astype(int)) 
    X,y = twospirals(50000)
    X = make_discrete(X,m=10)

    X = X.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, 
                                        test_size=0.3, 
                                        random_state=seed,
                                        stratify=y)
       
    # X_train, X_test, y_train, y_test = train_test_split(
    #                                     X, y, 
    #                                     test_size=0.3, 
    #                                     random_state=seed,
    #                                     stratify=y)
    pw = PazzaniWrapper(seed)
    transformer,features,model = pw.search(X_train,y_train)