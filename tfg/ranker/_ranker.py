
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import construct_features
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes 
from tfg.utils import symmetrical_uncertainty
from tqdm.autonotebook  import tqdm



class RankerLogicalFeatureConstructor(TransformerMixin,ClassifierMixin,BaseEstimator):
    """First proposal of a hybrid Ranker and Wrapper.

    Follows 3 steps to build a ranker based on symmetrical uncertainty (SU) of every possible logical feature of depth 1
    (1 operator, 2 operands), using XOR, AND and OR operator.
        - Find out combinations of values in database of every pair of features Xi, Xj:
            - Example: 
                - Xi = [1,2,3,2]
                - Xj = ['a','b','c','a']
                Possible combinations:
                    [(1,'a'),(2,'b'),(3,'c'),(2,'a')]
        - Apply operator to every combination:
            - Example: 
                - Xi = [1,2,3,2]
                - Xj = ['a','b','c','a']
                Possible combinations:
                    [(1,'a','AND'),(2,'b','AND'),(3,'c','AND'),(2,'a','AND'),
                    (1,'a','OR'),(2,'b','OR'),(3,'c','OR'),(2,'a','OR'),
                    (1,'a','XOR'),(2,'b','XOR'),(3,'c','XOR'),(2,'a','XOR')]
        - Add original variables to the list
        - Evaluate SU for every value in the list, and rank them
        - Go over the list following one of the two strategies proposed and evaluate 
          the subset based on a leave one out cross validation directly with the NaiveBayes classifier.

    Parameters
    ----------
    strategy : str {eager,skip}
        After the ranking is built if the eager strategy is chosen we stop considering attributes
        when there is no improvement from one iteration to the next
    
    block_size : int, default=1
        Number of features that are added in each iteration
    
    encode_data : boolean
        Whether or not to encode the received data. If set to false the classifier 
        expects data to be encoded with an ordinal encoder.

    verbose : {boolean,int}
        If set to true it displays information of the remaining time 
        and inside variables.
        
    operators : array-like, deafult = ("XOR","AND","OR")
        Operators used for the constructed features.

    max_features : int, deafult = inf
        Maximum number of features to include in the selected subset

    max_iterations : int, deafult = inf
        Maximum number of iterations in the wrapper step.
           
    Attributes
    ----------
    feature_encoder_ : CustomOrdinalFeatureEncoder or None
        Encodes data in ordinal way with unseen values handling if encode_data is set to True.
    
    class_encoder_ : LabelEncoder or None
        Encodes Data in ordinal way for the class if encode_data is set to True.

    all_feature_constructors: array-like
        List of FeatureConstructor objects with all the possible logical 
        features
    
    symmetrical_uncertainty_rank: array-like
        SU for every feature in all_feature_constructors

    rank : array-like
        Array of indexes corresponding to the sorted SU rank (in descending order).
    
    final_feature_constructors:
        Selected feature subset (list of constructors)

    classifier: NaiveBayes
        Classifier used in the wrapper and to perform predictions after fitting.

    """
    def __init__(self,
                 strategy="eager",
                 block_size=10,
                 encode_data=True,
                 n_intervals = 5,
                 verbose=0,
                 operators=("AND","OR","XOR"),
                 max_features = float("inf"),
                 max_iterations=float("inf"),
                 metric="accuracy",
                 use_initials=False,
                 max_err=0):
        self.strategy = strategy
        self.block_size = max(block_size,1)
        self.encode_data = encode_data
        self.verbose = verbose
        self.operators= operators
        self.max_features = max_features
        self.max_iterations = max_iterations
        self.n_intervals = n_intervals
        self.metric = metric
        self.max_err=max_err
        allowed_strategies = ("eager","skip")
        self.use_initials = use_initials
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown operator type: %s, expected one of %s." % (self.strategy, allowed_strategies))

    def fit(self,X,y):
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            self.feature_encoder_ = CustomOrdinalFeatureEncoder(n_intervals=self.n_intervals)
            self.class_encoder_ = CustomLabelEncoder()
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)

        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()

        check_X_y(X,y)
        self.all_feature_constructors = construct_features(X,operators=self.operators)
        if self.verbose:
            print(f"Total number of constructed features: {len(self.all_feature_constructors)}")
        self.all_feature_constructors.extend([DummyFeatureConstructor(j) for j in range(X.shape[1])])
        self.symmetrical_uncertainty_rank = []
        
        for feature_constructor in self.all_feature_constructors:
            feature = feature_constructor.transform(X)
            su = symmetrical_uncertainty(f1=feature.flatten(),f2=y)
            self.symmetrical_uncertainty_rank.append(su)
        self.rank = np.argsort(self.symmetrical_uncertainty_rank)[::-1] #Descending order
        self.filter_features(X,y)
        return self

    def predict(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier.predict(X,y)

        
    def predict_proba(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier.predict_proba(X,y)

    def score(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier.score(X,y)

    def filter_features(self,X,y):
        '''After the rank is built this perform the greedy wrapper search'''
        check_is_fitted(self)
        self.classifier = NaiveBayes(encode_data = False,n_intervals=self.n_intervals,metric=self.metric)
        current_score  = np.NINF
        first_iteration = True
        current_features = []
        current_data = None
        rank_iter = iter(self.rank)
        if self.use_initials:
           current_features = [DummyFeatureConstructor(j) for j in range(X.shape[1])]
           rank_iter = filter(lambda x: not isinstance(x,DummyFeatureConstructor), rank_iter)
           current_data = X.copy()
           current_score = self.classifier.leave_one_out_cross_val(X,y,fit=True)
        if self.verbose:
            print()
            progress_bar = tqdm(total=len(self.rank), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
        iteration=0
        iterations_without_improvements=0
        for feature_constructor_index in rank_iter:
            iteration+=1
            if self.verbose:
                progress_bar.set_postfix({"n_features": len(current_features), "score": current_score})
                progress_bar.update(1)
                progress_bar.refresh()
            new_X  = [self.all_feature_constructors[feature_constructor_index].transform(X)]
            selected_features = [self.all_feature_constructors[feature_constructor_index]]
            for _ in range(self.block_size-1):
                try:
                    index = next(rank_iter)
                    selected_features.append(self.all_feature_constructors[index])
                    new_X.append(self.all_feature_constructors[index].transform(X))
                    if self.verbose:
                            progress_bar.update(1)
                            progress_bar.refresh()
                except:
                    #Block size does not divide the number of elements in the rank
                    break
            
            new_X = np.concatenate(new_X,axis=1)
            if iteration==1 and not self.use_initials:
                current_data = new_X
                current_score = self.classifier.leave_one_out_cross_val(current_data,y,fit=True)
                current_features = selected_features
                first_iteration=False
                if self.max_iterations <= iteration or (len(current_features) + self.block_size) > self.max_features:
                        break
                continue
            data = np.concatenate([current_data,new_X],axis=1)
            self.classifier.add_features(new_X,y)
            score = self.classifier.leave_one_out_cross_val(data,y,fit=False)
            if score > current_score :
                current_score = score
                current_data = data
                current_features.extend(selected_features)
                iterations_without_improvements=0
            else:
                iterations_without_improvements+=1
                for feature_index_to_remove in range(data.shape[1], data.shape[1]-new_X.shape[1],-1):
                    self.classifier.remove_feature(feature_index_to_remove-1)
                if self.strategy=="eager" and self.max_err < iterations_without_improvements:
                    break # Stops as soon as no impovement
            
            if self.max_iterations <= iteration or (len(current_features) + self.block_size) > self.max_features:
                break
        if self.verbose:
            progress_bar.close()
            print(f"\nFinal number of included features: {len(current_features)} - Final Score: {current_score}")
        self.final_feature_constructors = current_features
        return self

    def transform(self,X,y):
        check_is_fitted(self)
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
            y = self.class_encoder_.transform(y)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()

        check_X_y(X,y)
        new_X = []
        for feature_constructor in self.final_feature_constructors:
            new_X.append(feature_constructor.transform(X))
        return np.concatenate(new_X,axis=1),y
