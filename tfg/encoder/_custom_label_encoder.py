import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

class CustomLabelEncoder(LabelEncoder):
    def __init__(self):
        super(CustomLabelEncoder,self).__init__()

    def fit(self,y):
        self.label_encoder_ = super().fit(y)
        self.unique_value_ = self.unique_value(self.label_encoder_.classes_)
        return self
        
    def transform(self,y):
        check_is_fitted(self)
        classes = self.label_encoder_.classes_
        idx = np.searchsorted(classes,y)
        idx[idx==classes.shape[0]] = 0
        mask = classes[idx] == y
        y_transformed = np.full(fill_value = classes.shape[0],shape=y.shape)
        y_transformed[mask] = idx[mask]
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self)
        classes = self.label_encoder_.classes_

        unseen = y==classes.shape[0]
        if np.any(unseen):
            mask = ~unseen
            y_transormed = np.empty(shape=y.shape,dtype=classes.dtype)
            y_transormed[mask] = super().inverse_transform(y[mask])
            y_transormed[~mask] = self.unique_value_
            return y_transormed
        return classes[y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def unique_value(self,classes):
        for i in range(classes.shape[0],-1,-1):
            n = np.array([i],dtype=classes.dtype)
            if (n[0] == classes).any():
                continue
            else:
                return n[0]