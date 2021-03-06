import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
class CustomOrdinalFeatureEncoderDict(TransformerMixin, BaseEstimator):
    def fit(self,X, y=None):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if X.dtype=="O":
            X = X.astype(str)
        self.n_features = X.shape[1]
        self.categories_ = [np.unique(X[:,j]) for j in range(self.n_features)]
        self.encoded_ = [ dict(zip(cat,range(len(cat)))) for cat in self.categories_]
        self.inverse_encoded_ = [ dict(zip(range(len(cat)),cat)) for cat in self.categories_]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
        return self
    
    def transform(self,X,y=None):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        X = X.copy()
        if X.dtype=="O":
            X = X.astype(str)
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = X[i,j]
                if val in self.encoded_[j]:
                    X[i,j] = self.encoded_[j][val]
                else:
                    X[i,j] = self.unknown_values_[j]            
        return X.astype(int)

    def inverse_transform(self,X,y=None):
        check_is_fitted(self)
        X_copy = np.empty(X.shape,dtype=self.categories_[0].dtype)
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = X[i,j]
                if val in self.inverse_encoded_[j]:
                    X_copy[i,j] = self.inverse_encoded_[j][val]
                else:
                    X_copy[i,j] = np.nan     
        return X_copy
   
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)


    # def transform_columns(self,X,categories):
    #     if isinstance(X,pd.DataFrame):
    #         X = X.to_numpy()
    #     X = X.copy()
    #     check_is_fitted(self)
    #     if len(categories) != X.shape[1]:
    #         raise Exception(f"Expected {categories} features, got {X.shape[1]} instead")
    #     if not all(index <self.n_features for index in categories) :
    #         raise ValueError(f"All values must be between 0 and {self.n_features}")
    #     for X_index,j in enumerate(categories):
    #         idx = np.searchsorted(self.sorted_categories_[j],X[:,X_index])
    #         idx[idx==self.sorted_encoded_[j].shape[0]] = 0
    #         mask = self.sorted_categories_[j][idx] == X[:,X_index]
    #         X[:,X_index]= np.where(mask , self.sorted_encoded_[j][idx],self.unknown_values_[j])
    #     return X.astype(int)

    # def add_features(self,X,transform=False,index=None):
    #     try:
    #         check_is_fitted(self)
    #     except NotFittedError as e:
    #         self.fit(X)
    #         if transform:
    #             return self.transform(X)
    #         return self
        
    #     self.n_features += X.shape[1]
    #     new_categories = [np.unique(X[:,j]) for j in range(X.shape[1])]
    #     if index is not None:
    #         sort_index = np.argsort(index)
    #         index_with_column = list(enumerate(index))
    #         for i in sort_index:
    #             column,list_insert_index = index_with_column[i]
    #             self.categories_.insert(list_insert_index,new_categories[column]) 
    #     else:
    #         self.categories_.extend(new_categories)
    #     self.sort_index_ = [cat.argsort() for cat in self.categories_]
    #     self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
    #     self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
    #     self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
    #     if transform:
    #         if index is None:
    #             index =  list(range(len(self.categories_)-len(new_categories),len(self.categories_)))
    #         return self.transform_columns(X,categories=index)
            
    # def remove_feature(self,index):
    #     check_is_fitted(self)
    #     self.n_features -=1 
    #     del self.categories_[index] 
    #     self.sort_index_ = [cat.argsort() for cat in self.categories_]
    #     self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
    #     self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
    #     self.unknown_values_ = [cat.shape[0] for cat in self.categories_]

