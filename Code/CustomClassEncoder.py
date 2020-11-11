import numpy as np
from sklearn.base import BaseEstimator, TransformerMixIn
from sklearn.utils.validation import check_is_fitted
from numba import njit
from numba.core import types
from numba.typed import Dict


class CustomClassEncoder(BaseEstimator, TransformerMixIn):
    def __init__(self):
        pass
    def fit(self):
        pass
    def transform(self):
        pass
    def fit_transform(self):
        pass