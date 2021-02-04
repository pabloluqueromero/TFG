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
def compute_total_probability_(class_values_count_,feature_values_count_,alpha):
    total_probability_ = class_values_count_ + alpha*feature_values_count_.reshape(-1,1)
    total_probability_ = np.where(total_probability_==0,1,total_probability_)
    total_probability_ = np.sum(np.log(total_probability_),axis=0)
    return total_probability_
    
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

        
        self.feature_values_count_per_element_ = [np.bincount(X[:,j]) for j in range(self.column_count_)]
        self.feature_values_count_ = np.array([(feature_counts).shape[0] for feature_counts in self.feature_values_count_per_element_])
        self.feature_unique_values_count_ = np.array([(feature_counts!=0).sum() for feature_counts in self.feature_values_count_per_element_])
        
        self._compute_probabilities(X, y)
        self._compute_independent_terms()
        return self

    def _compute_independent_terms(self):
        self.total_probability_ = compute_total_probability_(self.class_values_count_,self.feature_unique_values_count_,self.alpha)
        self.indepent_term_ = self.class_log_count_ - self.total_probability_

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
        probabilities += self.indepent_term_
        output = np.argmax(probabilities, axis=1)
        if self.encode_data:
            output = self.class_encoder_.inverse_transform(output)
        return output

    def predict_proba(self, X: np.ndarray):
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.probabilities_,self.feature_values_count_,self.alpha)
        probabilities += self.indepent_term_
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
    #         feature_values_count_per_element_ = []
    #         for j in range(X.shape[1]):
    #             feature_count = self.feature_values_count_per_element_[j].copy()
    #             if  example[j] < feature_count.shape[0]: #Could be unknown
    #                 feature_count[example[j]] -= (1 if feature_count[example[j]] else 0)
    #             feature_values_count_per_element_.append(feature_count)
    #         feature_values_count_ = np.array([(feature_counts!=0).sum() for feature_counts in feature_values_count_per_element_])
    #         total_probability_ = compute_total_probability_(class_values_count_,feature_values_count_, self.alpha)
    #         class_values_count_ = np.where(class_values_count_==0,1,class_values_count_)
    #         indepent_term_ = np.log(class_values_count_) - total_probability_

    #         probabilities_loo = []
    #         for col in range(X.shape[1]):
    #             probabilities_loo_j = self.probabilities_[col].copy()
    #             updated_value = np.clip(np.exp(probabilities_loo_j[example[col]][label])-1,a_min = self.alpha,a_max =None)
    #             updated_value = np.where(updated_value==0,1,updated_value)
    #             probabilities_loo_j[example[col]][label]= np.log(updated_value)
    #             probabilities_loo.append(probabilities_loo_j)
    #         prediction = _predict(np.array([example]),probabilities_loo,feature_values_count_,self.alpha)+ indepent_term_
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
            feature_values_count_per_element_ = self.feature_values_count_per_element_.copy()
            class_values_count_ = self.class_values_count_.copy()
            class_values_count_[label]-=1
            total_probability_ = compute_total_probability_(class_values_count_,self.feature_unique_values_count_, self.alpha)
            log_proba[i] -= total_probability_
            update_value = np.log(class_values_count_ + self.alpha)
            for j in range(X.shape[1]):
                p = self.probabilities_[j][example[j]] 
                log_proba[i] += p
                log_proba[i,label] -= p[label] 
                log_proba[i,label] += np.log(np.exp(p[label])-1)
                if feature_values_count_per_element_[j][example[j]] == 1:
                    log_proba[i] +=update_value
        prediction = np.argmax(log_proba ,axis=1)
        return np.sum(prediction == y)/y.shape[0]

    def add_features(self,X,y): 
        check_is_fitted(self)
        check_X_y(X,y)
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if self.encode_data:
            y = self.class_encoder_.transform(y) #y should the same than the one that was first fitted for now  ----> FUTURE IMPLEMENTATION
            X = self.feature_encoder_.add_features(X,transform=True)

        self.column_count_ += X.shape[1]
        new_feature_value_count_per_element =[np.bincount(X[:,j]) for j in range(X.shape[1])]
        self.feature_values_count_per_element_.extend(new_feature_value_count_per_element)
        new_feature_value_counts = np.array([(feature_counts).shape[0] for feature_counts in new_feature_value_count_per_element])
        self.feature_values_count_ = np.concatenate([self.feature_values_count_,new_feature_value_counts])
        new_probabilities = _get_probabilities(X,y,new_feature_value_counts,self.n_classes_,self.alpha)
        self.probabilities_.extend(new_probabilities)

        new_real_unique_feature_value_counts = np.array([(feature_counts!=0).sum() for feature_counts in new_feature_value_count_per_element])
        self.feature_unique_values_count_ = np.concatenate([self.feature_unique_values_count_,new_real_unique_feature_value_counts])
        feature_contribution = compute_total_probability_(self.class_values_count_,new_real_unique_feature_value_counts,self.alpha)
        self.total_probability_ +=  feature_contribution
        self.indepent_term_ -= feature_contribution
        return self

    
    def remove_feature(self,index):
        check_is_fitted(self)
        if self.column_count_ <=1:
            raise Exception("Cannot remove only feature from classifier")       
        if not 0 <= index <= self.column_count_:
            raise Exception(f"Feature index not valid, expected index between 0 and {self.column_count_}")       
        self.column_count_-=1
        
        feature_contribution = self.class_values_count_ + self.alpha*self.feature_unique_values_count_[index]
        feature_contribution = np.where(feature_contribution==0,1,feature_contribution)
        feature_contribution = np.log(feature_contribution)
        self.total_probability_ -=  feature_contribution
        self.indepent_term_ += feature_contribution

        self.feature_unique_values_count_ = np.delete(self.feature_unique_values_count_,index)
        self.feature_values_count_ = np.delete(self.feature_values_count_,index)
        del self.feature_values_count_per_element_[index]
        del self.probabilities_[index]
        
        if self.encode_data:
            self.feature_encoder_.remove_feature(index)
        return self

    def score(self, X: np.ndarray, y: np.ndarray):
        return np.sum(self.predict(X) == y)/y.shape[0]