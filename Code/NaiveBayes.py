import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from numba import jit

@jit(nopython=False)
def _get_probabilities(X:np.array, y:np.array, feature_values_count_:np.array, n_classes:int, alpha:float):
    probabilities = []
    for i in range(X.shape[1]):
        counts = _get_counts(X[:, i], y, feature_values_count_[i], n_classes)
        probabilities.append(np.log(counts+alpha))
    return probabilities

@jit(nopython=False)
def _get_counts(column:np.array, y:np.array, n_features:int, n_classes:int):
    counts = np.zeros((n_features, n_classes))
    for i in range(column.shape[0]):
        counts[column[i],y[i]]+=1
    return counts

@jit(nopython=False)
def _predict(X:np.array, probabilities:list):
    log_probability = np.zeros((X.shape[0], probabilities[0].shape[0]))
    for i in range(X.shape[1]):
        log_probability += probabilities[i][X[:,i],:]
    return log_probability


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1, encode_data=True):
        self.alpha = alpha
        self.encode_data = encode_data

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.row_count_, self.column_count_ = X.shape
        if self.encode_data:
            self.feature_encoder_ = OrdinalEncoder(dtype=np.intc)
            self.feature_encoder_.fit(X)
            self.class_encoder_ = LabelEncoder()
            self.class_encoder_.fit(y)
            X = self.feature_encoder_.transform(X)
            y = self.class_encoder_.transform(y)


        self.class_values_ = np.unique(y)
        self.class_values_count_ = np.bincount(y)
        self.n_classes_ = self.class_values_.shape[0]
        self.class_probabilities_ = self.class_values_count_/100

        self.feature_values_ = np.array([np.unique(column) for column in X.T])
        self.feature_values_count_ = np.array([feature.shape[0] for feature in self.feature_values_])
        self._compute_probabilities(X, y)
        self._compute_smoothness_correction()

    def _compute_smoothness_correction(self):
            term_1 = [np.log(count) for count in self.class_values_count_]
            term_3 = [-sum(np.log(self.class_values_count_[c] + self.alpha*self.feature_values_count_[f]) 
                            for f in range(self.column_count_)) 
                            for c in self.class_values_]
            self.smoothness_correction = np.array(term_1) + np.array(term_3)
            
    def _compute_probabilities(self, X: np.ndarray, y: np.ndarray):
        self.probabilities_=np.array(_get_probabilities(X, y, self.feature_values_count_, self.n_classes_, self.alpha))

    
    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X,self.probabilities_)
        probabilities += self.smoothness_correction
        output = np.argmax(probabilities, axis=1) 
        if self.encode_data:
            output=self.class_encoder_.inverse_transform(output)
        return output

    def score(self, X: np.ndarray, y: np.ndarray):
        if self.encode_data:
            y= self.class_encoder_.transform(y)
        return np.sum(self.predict(X) == y)/y.shape[0]