import numpy as np
        
class CustomFeatureEncoder:
    def __init__(self):
        self.is_fitted=False

    def fit(self,X, y=None):
        self.categories_ = [np.unique(column) for column in X.T]
        self.n_features = X.shape[1]
        self.is_fitted = True
        return self
    
    def transform(self,X,y=None):
        if not self.is_fitted:
            raise Exception(f"{self.__class__.__name__} instance not fitted")
        if self.n_features != X.shape[1]:
            raise Exception(f"Different number of attributes encountered while fitting")
        X_1 = np.zeros(shape=X.shape,dtype=np.int32)
        for j in range(X.shape[1]):
            mask = np.zeros(shape=X[:,j].shape[0],dtype=np.bool_)
            for c in range(self.categories_[j].shape[0]):
                mask_c = X[:,j] == self.categories_[j][c]
                mask = np.logical_or(mask_c, mask)
                X_1[:,j][mask_c]=c
            mask = np.logical_not(mask)
            X_1[:,j][mask] = self.categories_[j].shape[0]
        return X_1
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

if __name__ =="__main__":
    ce = CustomFeatureEncoder()
    X=np.array([
        ['P','+'],
        ['P2','-']
    ])
    X1=np.array([
        ['P','+'],
        ['P2','l'],
        ['i','l']
    ])
    ce.fit(X)
    print(ce.transform(X1))
            