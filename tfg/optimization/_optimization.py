
import pandas as pd
import numpy as np

from sklearn.utils.validation import check_is_fitted

class OptimizationMixIn:
    '''
    Mixing for transforming features in best_features for optimization algorithms
    '''
    def transform(self,X,y=None):
        check_is_fitted(self)
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
            if y is not None:
                y = self.class_encoder_.transform(y)
        X = np.concatenate([ f.transform(X) for f in self.best_features],axis=1)
        return X,y

    def predict(self,X):
        X, _ = self.transform(X)
        if self.encode_data:
            return self.class_encoder_.inverse_transform(self.classifier_.predict(X))
        return self.classifier_.predict(X)

        
    def predict_proba(self,X):
        X,_ = self.transform(X)
        return self.classifier_.predict_proba(X)

    def score(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier_.score(X,y)