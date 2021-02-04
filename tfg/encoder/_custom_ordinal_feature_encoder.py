import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
class CustomOrdinalFeatureEncoder(TransformerMixin, BaseEstimator):
    def fit(self,X, y=None):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        self.n_features = X.shape[1]
        self.categories_ = [np.unique(X[:,j]) for j in range(self.n_features)]
        self.sort_index_ = [cat.argsort() for cat in self.categories_]
        self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
        self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
        return self
    
    def transform(self,X,y=None):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.copy()
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")
        for j in range(X.shape[1]):
            idx = np.searchsorted(self.sorted_categories_[j],X[:,j])
            idx[idx==self.sorted_encoded_[j].shape[0]] = 0
            mask = self.sorted_categories_[j][idx] == X[:,j]
            X[:,j]= np.where(mask , self.sorted_encoded_[j][idx],self.unknown_values_[j])
        return X.astype(int)

    def inverse_transform(self,X,y=None):
        check_is_fitted(self)
        X_copy = np.empty(X.shape,dtype=self.categories_[0].dtype)
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")
        for j in range(X.shape[1]):
            inverse_idx = X[:,j]
            mask = inverse_idx==self.sorted_categories_[j].shape[0]
            inverse_idx[mask] = 0
            X_copy[:,j] = np.where(mask,np.nan,self.sorted_categories_[j][inverse_idx])
        return X_copy
    
    # def inverse_transform_columns(self,X,columns=None):
    #     check_is_fitted(self)
    #     if columns == None:
    #         columns = np.arange(X.shape[1])
    #     X_copy = np.empty(X[:,columns].shape,dtype=self.categories_[0].dtype)
    #     for j in columns:
    #         inverse_idx = X[:,j]
    #         mask = inverse_idx==self.sorted_categories_[j].shape[0]
    #         inverse_idx[mask] = 0
    #         X_copy[:,j] = np.where(mask,np.nan,self.sorted_categories_[j][inverse_idx])
    #     return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

    def get_index(self):
        check_is_fitted(self)
        return self.sorted_categories_

    def transform_columns(self,X,categories):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.copy()
        check_is_fitted(self)
        if len(categories) != X.shape[1]:
            raise Exception(f"Expected {categories} features, got {X.shape[1]} instead")
        if not all(index <self.n_features for index in categories) :
            raise ValueError(f"All values must be between 0 and {self.n_features}")
        for X_index,j in enumerate(categories):
            idx = np.searchsorted(self.sorted_categories_[j],X[:,X_index])
            idx[idx==self.sorted_encoded_[j].shape[0]] = 0
            mask = self.sorted_categories_[j][idx] == X[:,X_index]
            X[:,X_index]= np.where(mask , self.sorted_encoded_[j][idx],self.unknown_values_[j])
        return X.astype(int)

    def add_features(self,X,transform=False):
        try:
            check_is_fitted(self)
        except NotFittedError as e:
            self.fit(X)
            if transform:
                return self.transform(X)
            return self
        
        self.n_features +=X.shape[1]
        new_categories = [np.unique(X[:,j]) for j in range(X.shape[1])]
        self.categories_.extend(new_categories)
        self.sort_index_ = [cat.argsort() for cat in self.categories_]
        self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
        self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
        if transform:
            index =  list(range(len(self.categories_)-len(new_categories),len(self.categories_)))
            return self.transform_columns(X,categories=index)
    def remove_feature(self,index):
        check_is_fitted(self)
        self.n_features -=1 
        del self.categories_[index] 
        self.sort_index_ = [cat.argsort() for cat in self.categories_]
        self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
        self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]

