import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


class CustomLabelEncoder:
    """Label encoder

    Label encoder using numpy's searchsorted method to encode data to an integer ordinal representation.
    Automatic handling of unseen values (transformed to n, where n is the number of unique values)

    Parameters
    ----------

    Attributes
    ----------
    classes_ : int
        Categories sorted in alphanumeric order
    """

    def fit(self, y):
        self.classes_ = np.sort(pd.unique(y))
        return self

    def transform(self, y):
        check_is_fitted(self)
        classes = self.classes_
        idx = np.searchsorted(classes, y)
        idx[idx == classes.shape[0]] = 0
        mask = classes[idx] == y
        y_transformed = np.full(fill_value=classes.shape[0], shape=y.shape)
        y_transformed[mask] = idx[mask]
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self)
        idx = y == self.classes_.shape[0]
        if idx.any():
            unique_value_ = self.unique_value(self.classes_)
        else:
            unique_value_ = -1
        return np.where(y == self.classes_.shape[0], unique_value_, self.classes_[y])

    def fit_transform(self, y):
        self.classes_, y_transformed = np.unique(y, return_inverse=True)
        return y_transformed

    def unique_value(self, classes):
        possible_unique = np.arange(classes.shape[0]+1)
        return np.setdiff1d(possible_unique, classes)[0]
