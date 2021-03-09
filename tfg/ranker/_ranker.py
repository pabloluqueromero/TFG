
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.feature_construction import construct_features
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes 
from tfg.utils import symmetrical_uncertainty
from tqdm.autonotebook  import tqdm



class RankerLogicalFeatureConstructor(BaseEstimator,TransformerMixin):

    def __init__(self,strategy="eager",block_size=10,encode_data=True,verbose=0,operators=("AND","OR","XOR"),max_features = float("inf"),max_iterations=float("inf")):
        self.strategy = strategy
        self.block_size = max(block_size,1)
        self.encode_data = encode_data
        self.verbose = verbose
        self.operators= operators
        self.max_features = max_features
        self.max_iterations = max_iterations
        allowed_strategies = ("eager","skip")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown operator type: %s, expected one of %s." % (self.strategy, allowed_strategies))

    def fit(self,X,y):
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            self.feature_encoder_ = CustomOrdinalFeatureEncoder()
            self.class_encoder_ = LabelEncoder()
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
        current_score  = np.NINF
        first_iteration = True
        current_features = []
        current_data = None
        rank_iter = iter(self.rank)
        if self.verbose:
            print()
            progress_bar = tqdm(total=len(self.rank), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
        iteration=0
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
                    break
            
            new_X = np.concatenate(new_X,axis=1)
            if iteration==1:
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
            else:
                for feature_index_to_remove in range(data.shape[1], data.shape[1]-new_X.shape[1],-1):
                    self.classifier.remove_feature(feature_index_to_remove-1)
                if self.strategy=="eager":
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
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
            y = self.class_encoder_.transform(y)

        check_X_y(X,y)
        new_X = []
        for feature_constructor in self.final_feature_constructors:
            new_X.append(feature_constructor.transform(X))
        return np.concatenate(new_X,axis=1),y