
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.feature_construction import construct_features
from tfg.feature_construction import FeatureDummyConstructor
from tfg.naive_bayes import NaiveBayes 
from tfg.utils import symmetrical_uncertainty



class RankerLogicalFeatureConstructor(BaseEstimator,TransformerMixin):

    def __init__(self,strategy="eager",block_size=10,encode_data=True,verbose=0):
        self.strategy = strategy
        self.block_size = block_size
        self.encode_data = encode_data
        self.verbose = verbose
        allowed_strategies = ("eager","skip")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown operator type: %s, expected one of %s." % (self.strategy, allowed_strategies))

    def fit(self,X,y):
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
        self.all_feature_constructors = construct_features(X)
        if self.verbose:
            print(f"Total number of constructed features: {len(self.all_feature_constructors)}")
        self.all_feature_constructors.extend([FeatureDummyConstructor(j) for j in range(X.shape[1])])
        self.symmetrical_uncertainty_rank = []
        
        for feature_constructor in self.all_feature_constructors:
            feature = feature_constructor.transform(X)
            su = symmetrical_uncertainty(X=feature,y=y,f1=0)
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
        check_is_fitted(self)
        self.classifier = NaiveBayes(encode_data = False)
        rank_iter = iter(self.rank)
        current_score  = np.NINF
        first_iteration = True
        current_features = []
        for feature_constructor_index in rank_iter:
            if self.verbose:
                print(f"Current number of included features: {len(current_features)}   - Current Score: {current_score}")
            new_X  = [self.all_feature_constructors[feature_constructor_index].transform(X)]
            selected_features = [self.all_feature_constructors[feature_constructor_index]]
            for _ in range(self.block_size):
                try:
                    index = next(rank_iter)
                    selected_features.append(self.all_feature_constructors[index])
                    new_X.append(self.all_feature_constructors[index].transform(X))
                except:
                    break
            new_X = np.concatenate(new_X,axis=1)
            if first_iteration:
                current_score = self.classifier.leave_one_out_cross_val(new_X,y,fit=True)
                current_features = selected_features
            self.classifier.add_features(new_X,y)
            score = self.classifier.leave_one_out_cross_val(new_X,y,fit=True)
            if current_score > score:
                current_score = score
                current_features.extend(selected_features)
            else:
                for feature_index_to_remove in range(len(current_features),len(current_features)-len(new_X)):
                    self.classifier.remove_feature(feature_index_to_remove-1)
                break # Stops as soon as no impovement

        if self.verbose:
            print(f"Final number of included features: {len(current_features)} - Final Score: {current_score}")
        self.final_feature_constructors = current_features
        return self

    def transform(self,X,y):
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
        new_X = []
        for feature_constructor in self.final_feature_constructors:
            new_X.append(feature_constructor.transform(X))
        return np.concatenate(new_X,axis=1),y



        







