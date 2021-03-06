import math
import numpy as np
import pandas as pd

from numba import njit
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
#Local Imports
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.utils import get_scorer
  
#For unseen values we want that log(0) = -inf
np.seterr(divide='ignore')
"""
Enhanced methods with Numba nopython mode
"""
@njit
def _get_tables(X: np.array, y: np.array , n_classes: int, alpha: float):
    """Computes conditional log count for each value of each feature"""
    smoothed_log_counts = []
    smoothed_counts = []
    feature_values_count = []
    feature_values_count_per_element = []
    feature_unique_values_count = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        feature_values_count.append(np.max(feature)+1)
        counts = _get_counts(feature, y, feature_values_count[i], n_classes)
        feature_values_count_per_element.append(np.sum(counts,axis=1))
        feature_unique_values_count.append((feature_values_count_per_element[i]!=0).sum())
        
        smoothed_count = counts+alpha
        smoothed_counts.append(smoothed_count)

        smoothed_count_log = np.log(smoothed_count)
        smoothed_log_counts.append(smoothed_count_log)
    return smoothed_counts,smoothed_log_counts,np.array(feature_values_count),feature_values_count_per_element,np.array(feature_unique_values_count)

@njit
def _get_counts(column: np.ndarray, y: np.ndarray, n_features_: int, n_classes: int):
    """Computes count for each value of each feature for each class value"""
    counts = np.zeros((n_features_, n_classes),dtype=np.float64)
    for i in range(column.shape[0]):
        counts[column[i], y[i]] += 1
    return counts

@njit
def compute_total_probability_(class_count_,feature_values_count_,alpha):
    """Computes count for each value of each feature for each class value"""
    total_probability_ = class_count_ + alpha*feature_values_count_.reshape(-1,1)
    total_log_probability_ = np.log(total_probability_)
    total_log_probability_ = np.where(total_log_probability_==np.NINF,0,total_log_probability_)
    total_log_probability_ = np.sum(total_log_probability_,axis=0)
    return total_log_probability_
    
def _predict(X: np.ndarray, smoothed_log_counts_:np.ndarray, feature_values_count_:np.ndarray,alpha:float):
    """Computes the log joint probability"""
    log_probability = np.zeros((X.shape[0], smoothed_log_counts_[0].shape[1])) #(n_samples,n_classes)
    log_alpha= (np.log(alpha) if alpha else 0)
    for j in range(X.shape[1]):
        log_probability = _predict_single(log_probability,j,X,feature_values_count_,smoothed_log_counts_[j],log_alpha)
    return log_probability

@njit
def _predict_single(log_probability,j,X,feature_values_count_,smoothed_log_counts_,log_alpha):
    mask = X[:, j] < feature_values_count_[j] #Values known in the fitting stage
    index = X[:, j][mask]
    log_probability[mask,:] += smoothed_log_counts_[index]   # Only known values that are in probabilities
    mask = np.logical_not(mask)       
    log_probability[mask,:] += log_alpha   #Unknown values that are not in probabilities => log(0+alpha)
    return log_probability

class NaiveBayes(ClassifierMixin,BaseEstimator):
    """A Naive Bayes classifier.

    Simple NaiveBayes classifier accepting non-encoded input, enhanced with numba using MAP
    to predict most likely class.

    Parameters
    ----------
    alpha : {float, array-like}, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing). If it is an array it is 
        expected to have the same size as number of attributes

    encode_data : bool, default=True
        Encode data when data is not encoded by default with an OrdinalEncoder
    
    discretize : bool, default=True
        Discretize numerical data
    
    n_intervals : int or None, default=5
        Discretize numerical data using the specified number of intervals
    
    Attributes
    ----------
    feature_encoder_ : CustomOrdinalFeatureEncoder or None
        Encodes data in ordinal way with unseen values handling if encode_data is set to True.
    
    class_encoder_ : LabelEncoder or None
        Encodes Data in ordinal way for the class if encode_data is set to True.
    
    n_samples_ : int
        Number of samples  
    
    n_features_ : int
        Number of features
    
    n_classes_ : int
        Number of classes

    class_values_ : array-like of shape (n_classes_,)
        Array containing the values of the classes, as ordinal encoding is assumed it will be an array
        ranging from 0 to largest value for the class
    
    class_count_ : array-like of shape (n_classes_,)
        Array where `class_count_[i]` contains the count of the ith class value. 

    class_log_count_ : array-like of shape (n_classes_,)
        Array where `class_count_[i]` contains the log count of the ith class value. 
    
    feature_values_count_per_element_ : array-like of shape (column_count,~)
        Array where `feature_values_count_per_element_[i]` is an array where `feature_values_count_per_element_[i][j]`
        contains the count of the jth value for the ith feature. Assuming ordinal encoding, some values might be equal to 0
    
    feature_values_count_ : array-like of shape (column_count,)
        Array where `feature_values_count_per_element_[i]` is an integer with the number of possible values for the ith feature.

    feature_unique_values_count_ : array-like of shape (column_count,)
        Array where `feature_unique_values_count_[i]` is an integer with the number of unique seen values for the ith feature at
        fitting time. This is needed to compute the smoothing.

    total_probability_ : array-like of shape (n_classes,)
        Smoothing factor to be applied to the prediction. Array where `total_probability_[i]` if equal to
        class_count_[i] + alpha*feature_unique_values_count_
    
    self.indepent_term_ : array-like of shape (n_classes,)
        Independent term computed at fitting time. It includes the smoothing factor to be applied to the prediction and 
        the apriori probability.
    
    probabilities_ : array-like of shape (column_count,~)
        Array where `feature_values_count_per_element_[i]` is an array  of shape (where `feature_values_count_per_element_[i][j]`
        contains the count of the jth value for the ith feature. Assuming ordinal encoding, some values might be equal to 0
    """
    def __init__(self, alpha=1.0, encode_data=True, n_intervals=5,discretize=True,metric="accuracy"):
        self.alpha = alpha
        self.encode_data = encode_data
        self.n_intervals = n_intervals
        self.discretize = discretize
        self.metric = metric
        self._get_scorer()
        super().__init__()
    
    def _get_scorer(self):
        self.scorer = get_scorer(self.metric)
        if self.metric == "f1_score": #Unseen values for target class may cause errors
            self.scorer = lambda y_true,y_pred: get_scorer(self.metric)(y_true=y_true,
                                                                        y_pred=y_pred,
                                                                        average="macro", 
                                                                        zero_division=0)

    def set_params(self, **params):
        super().set_params(**params)
        self._get_scorer()
        
        
    def _compute_independent_terms(self):
        """Computes the terms that are indepent of the prediction"""
        self.total_probability_ = compute_total_probability_(self.class_count_,self.feature_unique_values_count_,self.alpha)
        # self.total_probability_ = compute_total_probability_(self.class_count_,self.feature_values_count_,self.alpha) #-->scikit uses this
        self.indepent_term_ = self.class_log_count_smoothed_ - self.total_probability_

    def _compute_class_counts(self, X: np.ndarray, y: np.ndarray):
        """Computes the counts for the priors"""
        self.n_classes_ = 1+np.max(y)
        self.class_count_ = np.bincount(y)
        self.class_log_count_ = np.log(self.class_count_)
        self.class_count_smoothed_ = (self.class_count_ + self.alpha)
        self.class_log_count_smoothed_ = np.log(self.class_count_smoothed_)


    def _compute_feature_counts(self, X: np.ndarray, y: np.ndarray):
        """Computes the conditional smoothed counts for each feature"""
        tables = _get_tables(
            X, y, self.n_classes_, self.alpha)
        self.smoothed_counts_ = tables[0]
        self.smoothed_log_counts_ = tables[1]
        self.feature_values_count_ = tables[2]
        self.feature_values_count_per_element_ = tables[3]
        self.feature_unique_values_count_ = tables[4]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fits the classifier with trainning data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features_)
            Training array that must be encoded unless
            encode_data is set to True

        y : array-like of shape (n_samples,)
            Label of the class associated to each sample.
            
        Returns
        -------
        self : object
        """
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            self.feature_encoder_ = CustomOrdinalFeatureEncoder(n_intervals = self.n_intervals, discretize= self.discretize)
            self.class_encoder_ = CustomLabelEncoder()
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        check_X_y(X,y)
        if X.dtype!=int:
            X = X.astype(int)
        if y.dtype!=int:
            y = y.astype(int)
        self.n_samples_, self.n_features_ = X.shape
        self._compute_class_counts(X, y)  
        self._compute_feature_counts(X, y)        
        self._compute_independent_terms()
        return self

    def predict(self, X: np.ndarray):
        """ Predicts the label of the samples based on the MAP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_)
           Training array that must be encoded unless
           encode_data is set to True

        Returns
        -------
        y : array-like of shape (n_samples)
            Predicted label for each sample.
        """
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if X.dtype!=int:
            X = X.astype(int)
        check_array(X)
        log_probabilities = _predict(X, self.smoothed_log_counts_,self.feature_values_count_,self.alpha)
        log_probabilities += self.indepent_term_
        output = np.argmax(log_probabilities, axis=1)
        if self.encode_data:
            output = self.class_encoder_.inverse_transform(output)
        return output

    def predict_proba(self, X: np.ndarray):
        """ Predicts the probability for each label of the samples based on the MAP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_)
           Training array that must be encoded unless
           encode_data is set to True

        Returns
        -------
        y : array-like of shape (n_classes,n_samples)
            Array where `y[i][j]` contains the MAP of the jth class for ith
            sample
        """
        check_is_fitted(self)
        if X.dtype!=int:
            X = X.astype(int)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        log_probabilities = _predict(X, self.smoothed_log_counts_,self.feature_values_count_,self.alpha)
        log_probabilities += self.indepent_term_
        log_prob_x = logsumexp(log_probabilities, axis=1)
        return np.exp(log_probabilities - np.atleast_2d(log_prob_x).T)

    def leave_one_out_cross_val(self,X,y,fit=True):
        """Efficient LOO computation"""
        if fit:
            self.fit(X,y)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
            y = self.class_encoder_.transform(y)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        
        if X.dtype!=int:
            X = X.astype(int)
        if y.dtype!=int:
            X = X.astype(int)
        log_alpha = np.log(self.alpha)
        log_proba = np.zeros((X.shape[0],self.n_classes_))
        for i in range(X.shape[0]):
            example, label = X[i], y[i]
            class_count_ = self.class_count_.copy()
            class_count_[label]-=1
            log_proba[i] = np.log(class_count_+self.alpha)
            for j in range(X.shape[1]):
                p = self.smoothed_log_counts_[j][example[j]].copy()
                p[label] = np.log(np.max([self.smoothed_counts_[j][example[j]][label]-1,self.alpha]))
                log_proba[i] += p
                if self.feature_values_count_per_element_[j][example[j]] == 1: 
                    update_value = np.log(class_count_ + (self.feature_unique_values_count_[j]-1)*self.alpha)
                else:
                    update_value  = np.log(class_count_ + (self.feature_unique_values_count_[j])*self.alpha)
                log_proba[i] -= np.where(update_value==np.NINF,0,update_value)
        y_pred = np.argmax(log_proba ,axis=1)
        return self.scorer(y,y_pred)

    def add_features(self,X,y,index=None): 
        """Updates classifier with new features

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_)
           Training array that must be encoded unless
           encode_data is set to True

        y : array-like of shape (n_samples,)
            Label of the class associated to each sample.
        
        index: {None,array-like of shape (X.shape[1])}
                Indicates where to insert each new feature, if it is None
                they are all appended at the very end.
        Returns
        -------
        self : object
        """
        check_is_fitted(self)
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            y = self.class_encoder_.transform(y) #y should be the same than the one that was first fitted for now  ----> FUTURE IMPLEMENTATION
            X = self.feature_encoder_.add_features(X,transform=True,index=index)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        check_X_y(X,y)
        if X.dtype!=int:
            X = X.astype(int)
        if y.dtype!=int:
            X = X.astype(int)
        
        self.n_features_ += X.shape[1]
        tables = _get_tables(X, y, self.n_classes_, self.alpha)
        new_smoothed_counts = tables[0]
        new_smoothed_log_counts = tables[1]
        new_feature_value_counts = tables[2]
        new_feature_value_counts_per_element = tables[3]
        new_feature_unique_values_count_ = tables[4]
        new_feature_contribution = compute_total_probability_(self.class_count_,new_feature_unique_values_count_,self.alpha)
        if index:
            sort_index = np.argsort(index)
            index_with_column = list(enumerate(index))
            for i in sort_index:
                column,list_insert_index = index_with_column[i]
                self.feature_values_count_per_element_.insert(list_insert_index,new_feature_value_counts_per_element[column])
                self.feature_values_count_ = np.insert(self.feature_values_count_,list_insert_index,new_feature_value_counts[column])
                self.smoothed_counts_.insert(list_insert_index,new_smoothed_counts[column])
                self.smoothed_log_counts_.insert(list_insert_index,new_smoothed_log_counts[column])
                self.feature_unique_values_count_ = np.insert(self.feature_unique_values_count_,list_insert_index,new_feature_unique_values_count_[column])
        else:
            self.feature_values_count_per_element_.extend(new_feature_value_counts_per_element)
            self.feature_values_count_ = np.concatenate([self.feature_values_count_,new_feature_value_counts])
            self.smoothed_counts_.extend(new_smoothed_counts)
            self.smoothed_log_counts_.extend(new_smoothed_log_counts)
            self.feature_unique_values_count_ = np.concatenate([self.feature_unique_values_count_,new_feature_unique_values_count_])

        self.total_probability_ +=  new_feature_contribution
        self.indepent_term_ -= new_feature_contribution
        
        return self

    
    def remove_feature(self,index):
        """Updates classifierby removing one feature (index)"""
        check_is_fitted(self)
        if self.n_features_ <=1:
            raise Exception("Cannot remove only feature from classifier")       
        if not 0 <= index < self.n_features_:
            raise Exception(f"Feature index not valid, expected index between 0 and {self.n_features_}")       
        self.n_features_-=1
        
        feature_contribution = self.class_count_ + self.alpha*self.feature_unique_values_count_[index]
        feature_contribution = np.log(feature_contribution)
        self.total_probability_ -=  feature_contribution
        self.indepent_term_ += feature_contribution

        self.feature_unique_values_count_ = np.delete(self.feature_unique_values_count_,index)
        self.feature_values_count_ = np.delete(self.feature_values_count_,index)
        del self.feature_values_count_per_element_[index]
        del self.smoothed_counts_[index]
        del self.smoothed_log_counts_[index]
        
        if self.encode_data:
            self.feature_encoder_.remove_feature(index)
        return self

    def score(self, X: np.ndarray, y: np.ndarray):
        """Computes the accuracy
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_)
           Training array that must be encoded unless
           encode_data is set to True

        y : array-like of shape (n_samples,)
            Label of the class associated to each sample.
        Returns
        -------
        score : float
                Percentage of correctly classified instances
        """
        y_pred = self.predict(X)
        return self.scorer(y,y_pred)