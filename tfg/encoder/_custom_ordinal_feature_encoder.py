import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer



warnings.filterwarnings('ignore',category=UserWarning)
class CustomOrdinalFeatureEncoder(TransformerMixin, BaseEstimator):
    def fit(self,X, y=None):
        X = X.copy()
        self.numerical_feature_index_ =  []
        if isinstance(X,pd.DataFrame):
            numerical_features = X.select_dtypes("float")
            if len(numerical_features.columns):
                self.discretizer = KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="quantile")
                X.loc[:,numerical_features.columns] = self.discretizer.fit_transform(numerical_features)
                X.loc[:,numerical_features.columns] = X.loc[:,numerical_features.columns].astype(int)
                self.numerical_feature_index_ =  list(X.columns.get_indexer(numerical_features.columns))
            X = X.to_numpy()
        X = X.astype(str)

        self.n_features = X.shape[1]
        self.categories_ = [np.unique(X[:,j]) for j in range(self.n_features)]
        self.sort_index_ = [cat.argsort() for cat in self.categories_]
        self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
        self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
        return self
    
    def transform(self,X,y=None):
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")
        X = X.copy()
        if isinstance(X,pd.DataFrame):
            numerical_features = X.select_dtypes("float")
            if len(numerical_features.columns):
                discretized_features = self.discretizer.transform(numerical_features)
                X.loc[:,numerical_features.columns] = discretized_features
                X.loc[:,numerical_features.columns] = X.loc[:,numerical_features.columns].astype(int)
            X = X.to_numpy()
        X = X.astype(str)
        
        X_copy = np.empty(shape=X.shape,dtype=int)
        for j in range(X_copy.shape[1]):
            idx = np.searchsorted(self.sorted_categories_[j],X[:,j])
            idx[idx==self.sorted_encoded_[j].shape[0]] = 0
            mask = self.sorted_categories_[j][idx] == X[:,j]
            X_copy[:,j]= np.where(mask , self.sorted_encoded_[j][idx],self.unknown_values_[j])
        return X_copy.astype(int)

    def inverse_transform(self,X,y=None):
        '''Inverse transform (numerical features cannot be restored)'''
        check_is_fitted(self)
        X_copy = np.empty(X.shape,dtype=self.categories_[0].dtype)
        check_is_fitted(self)
        if self.n_features != X.shape[1]:
            raise Exception(f"Expected {self.n_features} features, got {X.shape[1]} instead")
        for j in range(X_copy.shape[1]):
            inverse_idx = X[:,j]
            mask = inverse_idx==self.sorted_categories_[j].shape[0]
            inverse_idx[mask] = 0
            X_copy[:,j] = np.where(mask,np.nan,self.sorted_categories_[j][inverse_idx])
        return X_copy
    
    '''Translation method to be implemented and corrected not needed for the moment'''
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
        '''Method used to transform new columns wwhen dinamiccally adding features to the transformer.
           It is assumed that variables that need to be discretized have been taken cared of in previous
           steps'''
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        
        if X.dtype=="O":
            X = X.astype(str)
        X_copy = np.empty(shape=X.shape,dtype=int)
        check_is_fitted(self)

        if len(categories) != X.shape[1]:
            raise Exception(f"Expected {categories} features, got {X.shape[1]} instead")
        if not all(index <self.n_features for index in categories) :
            raise ValueError(f"All values must be between 0 and {self.n_features}")
        for j in range(X.shape[1]):
            cat = categories[j]
            idx = np.searchsorted(self.sorted_categories_[cat],X[:,j])
            idx[idx==self.sorted_encoded_[cat].shape[0]] = 0
            mask = self.sorted_categories_[cat][idx] == X[:,j]
            X_copy[:,j]= np.where(mask , self.sorted_encoded_[cat][idx],self.unknown_values_[cat])
        return X_copy

    def add_features(self,X,transform=False,index=None):
        try:
            check_is_fitted(self)
        except NotFittedError as e:
            self.fit(X)
            if transform:
                return self.transform(X)
            return self
        X = X.copy()
        if isinstance(X,pd.DataFrame):
            numerical_features = X.select_dtypes("float")
            if len(numerical_features.columns):
                temp_discretizer = KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="quantile")
                X.loc[:,numerical_features.columns] = temp_discretizer.fit_transform(numerical_features)
                X.loc[:,numerical_features.columns] = X.loc[:,numerical_features.columns].astype(int)
                new_index = X.columns.get_indexer(numerical_features.columns)
                if index:
                    index_with_column = list(enumerate(index))
                    numerical_index = np.array(index)[new_index]
                    sort_index = np.argsort(numerical_index)
                    numerical_index_with_column = [index_with_column[i] for i in numerical_index] 
                    last = 0 
                    for i in sort_index:
                        feature,list_insert_index = numerical_index_with_column[i]
                        while last <= len(self.numerical_feature_index_) and list_insert_index < self.numerical_feature_index_[last] :
                            last+=1
                        self.numerical_feature_index_.insert(last,list_insert_index)
                        self.discretizer.n_bins_= np.insert( self.discretizer.n_bins_,last,temp_discretizer.n_bins_[i],axis=1)
                        self.discretizer.bin_edges_= np.insert( self.discretizer.n_bins_,last,temp_discretizer.bin_edges_[i],axis=1)
                        for n in range(last+1,len(self.numerical_feature_index_)):
                            self.numerical_feature_index_[n]+=1
                        
                else:
                    new_index = X.columns.get_indexer(numerical_features.columns)+self.n_features
                    self.numerical_feature_index_.extend(new_index)
                    self.discretizer.n_bins_ = np.concatenate([self.discretizer.n_bins_,temp_discretizer.n_bins_],axis=1)
                    self.discretizer.bin_edges_ = np.concatenate([self.discretizer.bin_edges_,temp_discretizer.bin_edges_],axis=1)
            X = X.to_numpy()
        X = X.astype(str)
        self.n_features += X.shape[1]
        new_categories = [np.unique(X[:,j]) for j in range(X.shape[1])]
        if index is not None:
            sort_index = np.argsort(index)
            index_with_column = list(enumerate(index))
            for i in sort_index:
                column,list_insert_index = index_with_column[i]
                self.categories_.insert(list_insert_index,new_categories[column]) 
        else:
            self.categories_.extend(new_categories)
        self.sort_index_ = [cat.argsort() for cat in self.categories_]
        self.sorted_categories_ = [self.categories_[j][self.sort_index_[j]] for j in range(self.n_features)]
        self.sorted_encoded_ = [np.arange(self.categories_[j].shape[0])[self.sort_index_[j]] for j in range(self.n_features)]
        self.unknown_values_ = [cat.shape[0] for cat in self.categories_]
        if transform:
            if index is None:
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
        
        #Remove if it is a continuos feature
        try:
            i = self.numerical_feature_index_.index(index)
            del self.numerical_feature_index_[i]
            self.discretizer.bin_edges_ = np.delete(self.discretizer.bin_edges,i,axis=1)
            self.discretizer.n_bin_ = np.delete(self.discretizer.bin_edges,i,axis=1)
        except ValueError:
            pass


    def inverse_transform_element(self,feature_index,value):
        check_is_fitted(self)
        try:
            value = self.sorted_categories_[feature_index][value]
            try:
                i = self.numerical_feature_index_.index(feature_index)
                if int(value)==0:
                    value = [np.NINF,self.discretizer.bin_edges_[i][1]]
                elif int(value)==len(self.discretizer.bin_edges_[i])-2:
                    value = [self.discretizer.bin_edges_[i][int(value)],np.Inf]
                else:
                    value = self.discretizer.bin_edges_[i][int(value):int(value)+2]
            except:
                pass
            return value
        except:
            return np.nan