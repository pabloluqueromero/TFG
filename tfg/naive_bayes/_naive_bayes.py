import math
import numpy as np
import pandas as pd

from numba import njit
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

#Local Imports
from tfg.encoder import CustomOrdinalFeatureEncoder
  
"""
Enhanced methods with Numba nopython mode
"""
@njit
def _get_tables(X: np.array, y: np.array , n_classes: int, alpha: float):
    """Computes conditional log count for each value of each feature"""
    smoothed_log_counts = []
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
        smoothed_count = np.where(smoothed_count==0,np.NINF,smoothed_count)
        smoothed_log_counts.append(np.log(smoothed_count))
    return smoothed_log_counts,np.array(feature_values_count),feature_values_count_per_element,np.array(feature_unique_values_count)

@njit
def _get_counts(column: np.ndarray, y: np.ndarray, n_features: int, n_classes: int):
    """Computes count for each value of each feature for each class value"""
    counts = np.zeros((n_features, n_classes))
    for i in range(column.shape[0]):
        counts[column[i], y[i]] += 1
    return counts

@njit
def compute_total_probability_(class_values_count_,feature_values_count_,alpha):
    """Computes count for each value of each feature for each class value"""
    total_probability_ = class_values_count_ + alpha*feature_values_count_.reshape(-1,1)
    total_probability_ = np.where(total_probability_==0,np.NINF,total_probability_)
    total_probability_ = np.sum(np.log(total_probability_),axis=0)
    return total_probability_
    
def _predict(X: np.ndarray, probabilities:np.ndarray, feature_values_count_:np.ndarray,alpha:float):
    """Computes the log joint probability"""
    log_probability = np.zeros((X.shape[0], probabilities[0].shape[1]))
    log_alpha=(np.log(alpha) if alpha else 0)
    for j in range(X.shape[1]):
        log_probability = _predict_single(log_probability,j,X,feature_values_count_,probabilities[j],log_alpha)
    return log_probability

@njit
def _predict_single(log_probability,j,X,feature_values_count_,probabilities,log_alpha):
    mask = X[:, j] < feature_values_count_[j] #Values known in the fitting stage
    index = X[:, j][mask]
    log_probability[mask,:] += probabilities[index]   # Only known values that are in probabilities
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

    encode_data : bool, default=Ture
        Encode data when data is not encoded by default with an OrdinalEncoder
    
    Attributes
    ----------
    feature_encoder_ : CustomOrdinalFeatureEncoder or None
        Encodes data in ordinal way with unseen values handling if encode_data is set to True.
    
    class_encoder_ : LabelEncoder or None
        Encodes Data in ordinal way for the class if encode_data is set to True.
    
    row_count_ : int
        Number of samples  
    
    column_count_ : int
        Number of features
    
    n_classes_ : int
        Number of classes

    class_values_ : array-like of shape (n_classes_,)
        Array containing the values of the classes, as ordinal encoding is assumed it will be an array
        ranging from 0 to largest value for the class
    
    class_values_count_ : array-like of shape (n_classes_,)
        Array where `class_values_count_[i]` contains the count of the ith class value. 

    class_log_count_ : array-like of shape (n_classes_,)
        Array where `class_values_count_[i]` contains the log count of the ith class value. 
    
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
        class_values_count_[i] + alpha*feature_unique_values_count_
    
    self.indepent_term_ : array-like of shape (n_classes,)
        Independent term computed at fitting time. It includes the smoothing factor to be applied to the prediction and 
        the apriori probability.
    
    probabilities_ : array-like of shape (column_count,~)
        Array where `feature_values_count_per_element_[i]` is an array  of shape (where `feature_values_count_per_element_[i][j]`
        contains the count of the jth value for the ith feature. Assuming ordinal encoding, some values might be equal to 0
    """
    def __init__(self, alpha=1.0, encode_data=True):
        self.alpha = alpha
        self.encode_data = encode_data
        super().__init__()


    def _compute_independent_terms(self):
        """Computes the terms that are indepent of the prediction"""
        self.total_probability_ = compute_total_probability_(self.class_values_count_,self.feature_unique_values_count_,self.alpha)
        self.indepent_term_ = self.class_log_count_ - self.total_probability_

    def _compute_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Computes the conditional probabilities for each value of each feature"""
        tables = _get_tables(
            X, y, self.n_classes_, self.alpha)
        self.smoothed_log_counts_ = tables[0]
        self.feature_values_count_ = tables[1]
        self.feature_values_count_per_element_ = tables[2]
        self.feature_unique_values_count_ = tables[3]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fits the classifier with trainning data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training array that must be encoded unless
            encode_data is set to True

        y : array-like of shape (n_samples,)
            Label of the class associated to each sample.
            
        Returns
        -------
        self : object
        """
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            self.feature_encoder_ = CustomOrdinalFeatureEncoder()
            self.class_encoder_ = LabelEncoder()
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)

        check_X_y(X,y)
        self.row_count_, self.column_count_ = X.shape
        self.class_values_ = np.arange(0,1+np.max(y)) #Not needed
        self.class_values_count_ = np.bincount(y)
        self.class_log_count_ = np.log(self.class_values_count_,where = self.class_values_count_!=0)
        self.n_classes_ = self.class_values_.shape[0]

        
        # self.feature_values_count_per_element_ = [np.bincount(X[:,j]) for j in range(self.column_count_)]
        self._compute_probabilities(X, y)
        # self.feature_values_count_ = np.array([(feature_counts).shape[0] for feature_counts in self.feature_values_count_per_element_])
        # self.feature_unique_values_count_ = np.array([(feature_counts!=0).sum() for feature_counts in self.feature_values_count_per_element_])
        
        self._compute_independent_terms()
        return self

    def predict(self, X: np.ndarray):
        """ Predicts the label of the samples based on the MAP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Training array that must be encoded unless
           encode_data is set to True

        Returns
        -------
        y : array-like of shape (n_samples)
            Predicted label for each sample.
        """
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.smoothed_log_counts_,self.feature_values_count_,self.alpha)
        probabilities += self.indepent_term_
        output = np.argmax(probabilities, axis=1)
        if self.encode_data:
            output = self.class_encoder_.inverse_transform(output)
        return output

    def predict_proba(self, X: np.ndarray):
        """ Predicts the probability for each label of the samples based on the MAP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Training array that must be encoded unless
           encode_data is set to True

        Returns
        -------
        y : array-like of shape (n_classes,n_samples)
            Array where `y[i][j]` contains the MAP of the jth class for ith
            sample
        """
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.smoothed_log_counts_,self.feature_values_count_,self.alpha)
        probabilities += self.indepent_term_
        log_prob_x = logsumexp(probabilities, axis=1)
        return np.exp(probabilities - np.atleast_2d(log_prob_x).T)

    def leave_one_out_cross_val(self,X,y,fit=True):
        """Efficient LOO computation"""
        if fit:
            self.fit(X,y)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
            y = self.class_encoder_.transform(y)
        log_proba = np.zeros((X.shape[0],self.class_log_count_.shape[0]))
        log_proba+= self.class_log_count_
        for v in np.unique(y):
            log_proba[y==v,v] -= self.class_log_count_[v]
            log_proba[y==v,v] += np.log(self.class_values_count_[v]-1) if self.class_values_count_[v] >1 else np.NINF #Can't predict an unseen label
        for i in range(X.shape[0]):
            example, label = X[i], y[i]
            feature_values_count_per_element_ = self.feature_values_count_per_element_.copy()
            class_values_count_ = self.class_values_count_.copy()
            class_values_count_[label]-=1
            total_probability_ = compute_total_probability_(class_values_count_,self.feature_unique_values_count_, self.alpha)
            log_proba[i] -= total_probability_
            update_value = np.log(class_values_count_ + self.alpha)
            for j in range(X.shape[1]):
                p = self.smoothed_log_counts_[j][example[j]] 
                log_proba[i] += p
                log_proba[i,label] -= p[label] 
                log_proba[i,label] += np.log(np.exp(p[label])-1)
                if feature_values_count_per_element_[j][example[j]] == 1:
                    log_proba[i] +=update_value
        prediction = np.argmax(log_proba ,axis=1)
        return np.sum(prediction == y)/y.shape[0]

    def add_features(self,X,y,index=None): 
        """Updates classifier with new features

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if self.encode_data:
            y = self.class_encoder_.transform(y) #y should the same than the one that was first fitted for now  ----> FUTURE IMPLEMENTATION
            X = self.feature_encoder_.add_features(X,transform=True,index=index)
        check_X_y(X,y)
        
        self.column_count_ += X.shape[1]
        new_feature_value_count_per_element =[np.bincount(X[:,j]) for j in range(X.shape[1])]
        new_feature_value_counts = np.array([(feature_counts).shape[0] for feature_counts in new_feature_value_count_per_element])
        new_probabilities = _get_tables(X,y,new_feature_value_counts,self.n_classes_,self.alpha)
        new_real_unique_feature_value_counts = np.array([(feature_counts!=0).sum() for feature_counts in new_feature_value_count_per_element])
        feature_contribution = compute_total_probability_(self.class_values_count_,new_real_unique_feature_value_counts,self.alpha)
        if index:
            sort_index = np.argsort(index)
            index_with_column = list(enumerate(index))
            for i in sort_index:
                column,list_insert_index = index_with_column[i]
                self.feature_values_count_per_element_.insert(list_insert_index,new_feature_value_count_per_element[column])
                self.feature_values_count_ = np.insert(self.feature_values_count_,list_insert_index,new_feature_value_counts[column])
                self.smoothed_log_counts_.insert(list_insert_index,new_probabilities[column])
                self.feature_unique_values_count_ = np.insert(self.feature_unique_values_count_,list_insert_index,new_real_unique_feature_value_counts[column])
        else:
            self.feature_values_count_per_element_.extend(new_feature_value_count_per_element)
            self.feature_values_count_ = np.concatenate([self.feature_values_count_,new_feature_value_counts])
            self.smoothed_log_counts_.extend(new_probabilities)
            self.feature_unique_values_count_ = np.concatenate([self.feature_unique_values_count_,new_real_unique_feature_value_counts])

        self.total_probability_ +=  feature_contribution
        self.indepent_term_ -= feature_contribution
        
        return self

    
    def remove_feature(self,index):
        """Updates classifierby removing one feature (index)"""
        check_is_fitted(self)
        if self.column_count_ <=1:
            raise Exception("Cannot remove only feature from classifier")       
        if not 0 <= index <= self.column_count_:
            raise Exception(f"Feature index not valid, expected index between 0 and {self.column_count_}")       
        self.column_count_-=1
        
        feature_contribution = self.class_values_count_ + self.alpha*self.feature_unique_values_count_[index]
        feature_contribution = np.where(feature_contribution==0,np.NINF,feature_contribution)
        feature_contribution = np.log(feature_contribution)
        self.total_probability_ -=  feature_contribution
        self.indepent_term_ += feature_contribution

        self.feature_unique_values_count_ = np.delete(self.feature_unique_values_count_,index)
        self.feature_values_count_ = np.delete(self.feature_values_count_,index)
        del self.feature_values_count_per_element_[index]
        del self.smoothed_log_counts_[index]
        
        if self.encode_data:
            self.feature_encoder_.remove_feature(index)
        return self

    def score(self, X: np.ndarray, y: np.ndarray):
        """Computes the accuracy
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Training array that must be encoded unless
           encode_data is set to True

        y : array-like of shape (n_samples,)
            Label of the class associated to each sample.
        Returns
        -------
        score : float
                Percentage of correctly classified instances
        """
        return np.sum(self.predict(X) == y)/y.shape[0]