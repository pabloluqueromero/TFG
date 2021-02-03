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
  
'''
Enhanced methods with Numba nopython mode
'''
@njit
def _get_probabilities(X: np.array, y: np.array, feature_values_count_: np.array, n_classes: int, alpha: float):
    probabilities = []
    for i in range(X.shape[1]):
        smoothed_counts = _get_counts(X[:, i], y, feature_values_count_[i], n_classes) +alpha
        smoothed_counts = np.where(smoothed_counts==0,1,smoothed_counts)
        probabilities.append(np.log(smoothed_counts))
    return probabilities

@njit
def _get_counts(column: np.ndarray, y: np.ndarray, n_features: int, n_classes: int):
    counts = np.zeros((n_features, n_classes))
    for i in range(column.shape[0]):
        counts[column[i], y[i]] += 1
    return counts

@njit
def compute_total_probability(class_values_count,feature_values_count_,alpha):
    total_probability = class_values_count + alpha*feature_values_count_.reshape(-1,1)
    total_probability = np.where(total_probability==0,1,total_probability)
    total_probability = np.sum(np.log(total_probability),axis=0)
    return total_probability
    
#probabilities is not a squared matrix only solution would be to pad with zeros, inneficient when there are many features
def _predict(X: np.ndarray, probabilities:np.ndarray, feature_values_count_:np.ndarray,alpha:float):
    log_probability = np.zeros((X.shape[0], probabilities[0].shape[1]))
    log_alpha=(np.log(alpha) if alpha else 0)
    for j in range(X.shape[1]):
        mask = X[:, j] < feature_values_count_[j] #Values known in the fitting stage
        index = X[:, j][mask]
        log_probability[mask,:] += probabilities[j][index]   # Only known values that are in probabilities
        mask = np.logical_not(mask)       
        log_probability[mask,:] += log_alpha   #Unknown values that are not in probabilities => log(0+alpha)
    return log_probability

class NaiveBayes(ClassifierMixin,BaseEstimator):
    '''Simple NaiveBayes classifier accepting non-encoded input and enhanced with numba
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    encode_data: bool, default=Ture
        Encode data when data is not encoded by default with an OrdinalEncoder
    '''
    def __init__(self, alpha=1.0, encode_data=True):
        self.alpha = alpha
        self.encode_data = encode_data
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray):
        ''' Fit the classifier
            Attributes
            ----------
            feature_encoder_ : TransformerMixin
                    Encodes data in ordinal way with unknown values handling.
            class_encoder_: TransformerMixin
                    Encodes Data in ordinal way for the class.
            row_count_: int
                    Size of the dataset  
            column_count_: int
                    Number of features
            n_classes_: int
                    Number of classes
            feature_values_: int
                    Values of the clase 
            feature_values_count_: int
                    Values of the clase 
        '''
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
        self.class_values_ = np.arange(0,1+np.max(y))
        self.class_values_count_ = np.bincount(y)
        self.class_log_count_ = np.log(self.class_values_count_,where = self.class_values_count_!=0)
        self.n_classes_ = self.class_values_.shape[0]

        
        self.feature_values_count_per_element = [np.bincount(X[:,j]) for j in range(self.column_count_)]
        self.feature_values_count_ = np.array([(feature_counts).shape[0] for feature_counts in self.feature_values_count_per_element])
        self.feature_values_ = np.array([np.arange(j.shape[0]) for j in self.feature_values_count_per_element],dtype="object")
        
        self._compute_probabilities(X, y)
        self._compute_independent_terms()
        return self

    def _compute_independent_terms(self):
        real_unique_value_counts = np.array([(feature_counts!=0).sum() for feature_counts in self.feature_values_count_per_element])
        self.total_probability = compute_total_probability(self.class_values_count_,real_unique_value_counts,self.alpha)
        self.indepent_term = self.class_log_count_ - self.total_probability

    def _compute_probabilities(self, X: np.ndarray, y: np.ndarray):
        self.probabilities_ = _get_probabilities(
            X, y, self.feature_values_count_, self.n_classes_, self.alpha)

    def predict(self, X: np.ndarray):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.probabilities_,self.feature_values_count_,self.alpha)
        probabilities += self.indepent_term
        output = np.argmax(probabilities, axis=1)
        if self.encode_data:
            output = self.class_encoder_.inverse_transform(output)
        return output

    def predict_proba(self, X: np.ndarray):
        check_is_fitted(self)
        if self.encode_data:
            check_is_fitted(self.feature_encoder_)
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.probabilities_,self.feature_values_count_,self.alpha)
        probabilities += self.indepent_term
        log_prob_x = logsumexp(probabilities, axis=1)
        return np.exp(probabilities - np.atleast_2d(log_prob_x).T)

    # Less efficient
    # def leave_one_out_cross_val(self,X,y):
    #     self.fit(X,y)
    #     if isinstance(X,pd.DataFrame):
    #         X = X.to_numpy()
    #     if self.encode_data:
    #         X = self.feature_encoder_.transform(X)
    #         y = self.class_encoder_.transform(y)
    #     score = []
    #     for i in range(X.shape[0]):
    #         example, label = X[i], y[i]
    #         class_values_count_ = self.class_values_count_.copy()
    #         class_values_count_[label]-=(1 if class_values_count_[label] else 0)
    #         feature_values_count_per_element = []
    #         for j in range(X.shape[1]):
    #             feature_count = self.feature_values_count_per_element[j].copy()
    #             if  example[j] < feature_count.shape[0]: #Could be unknown
    #                 feature_count[example[j]] -= (1 if feature_count[example[j]] else 0)
    #             feature_values_count_per_element.append(feature_count)
    #         feature_values_count_ = np.array([(feature_counts!=0).sum() for feature_counts in feature_values_count_per_element])
    #         total_probability = compute_total_probability(class_values_count_,feature_values_count_, self.alpha)
    #         class_values_count_ = np.where(class_values_count_==0,1,class_values_count_)
    #         indepent_term = np.log(class_values_count_) - total_probability

    #         probabilities_loo = []
    #         for col in range(X.shape[1]):
    #             probabilities_loo_j = self.probabilities_[col].copy()
    #             updated_value = np.clip(np.exp(probabilities_loo_j[example[col]][label])-1,a_min = self.alpha,a_max =None)
    #             updated_value = np.where(updated_value==0,1,updated_value)
    #             probabilities_loo_j[example[col]][label]= np.log(updated_value)
    #             probabilities_loo.append(probabilities_loo_j)
    #         prediction = _predict(np.array([example]),probabilities_loo,feature_values_count_,self.alpha)+ indepent_term
    #         prediction = np.argmax( prediction,axis=1)
    #         score.append(prediction[0]==label)
    #     return np.mean(score)

    def leave_one_out_cross_val(self,X,y):
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
            log_proba[y==v,v] += np.log(self.class_values_count_[v]-1) if self.class_values_count_[v] >1 else -float("inf") #Can't predict an unseen label
        for i in range(X.shape[0]):
            example, label = X[i], y[i]
            feature_values_count_per_element = self.feature_values_count_per_element.copy()
            class_values_count_ = self.class_values_count_.copy()
            class_values_count_[label]-=1
            for j in range(X.shape[1]):
                p = self.probabilities_[j][example[j]] 
                log_proba[i] += p
                log_proba[i,label] -= p[label] 
                log_proba[i,label] += np.log(np.exp(p[label])-1)
                feature_values_count_per_element[j][example[j]]-=1
            feature_values_count_ = np.array([(feature_counts!=0).sum() for feature_counts in feature_values_count_per_element])
            total_probability = compute_total_probability(class_values_count_,feature_values_count_, self.alpha)
            log_proba[i] -=total_probability
        prediction = np.argmax(log_proba ,axis=1)
        return np.sum(prediction == y)/y.shape[0]

    def score(self, X: np.ndarray, y: np.ndarray):
        return np.sum(self.predict(X) == y)/y.shape[0]