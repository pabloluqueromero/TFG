import numpy as np
from sklearn.preprocessing import LabelEncoder
from CustomFeatureEncoder import CustomFeatureEncoder
from numba import njit

'''
Enhanced methods with Numba nopython mode
'''
@njit
def _get_probabilities(X: np.array, y: np.array, feature_values_count_: np.array, n_classes: int, alpha: float):
    probabilities = []
    for i in range(X.shape[1]):
        counts = _get_counts(X[:, i], y, feature_values_count_[i], n_classes)
        probabilities.append(np.log(counts+alpha))
    return probabilities


@njit
def _get_counts(column: np.array, y: np.array, n_features: int, n_classes: int):
    counts = np.zeros((n_features, n_classes))
    for i in range(column.shape[0]):
        counts[column[i], y[i]] += 1
    return counts

@njit
def _predict(X: np.array, probabilities: list,len_feature_values:list,alpha:float):
    log_probability = np.zeros((X.shape[0], probabilities[0].shape[0]))
    log_alpha=np.log(alpha)
    for j in range(X.shape[1]):
        mask = X[:, j] < len_feature_values[j] #Values known in the fitting stage
        index = X[:, j][mask]
        log_probability[mask,:] += probabilities[j][index, :]   #Known values that are in probabilities
        mask = np.logical_not(mask)       
        log_probability[mask,:] += log_alpha    #Unknown values that are not in probabilities => og(0+alpha)
    return log_probability

class NaiveBayes:
    def __init__(self, alpha=1, encode_data=True):
        self.alpha = alpha
        self.encode_data = encode_data
        self.is_fitted=False

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.row_count_, self.column_count_ = X.shape
        if self.encode_data:
            self.feature_encoder_ = CustomFeatureEncoder()#CustomFeatureEncoder()
            self.class_encoder_ = LabelEncoder()
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)

        self.class_values_ = np.unique(y)
        self.class_values_count_ = np.bincount(y)
        self.n_classes_ = self.class_values_.shape[0]
        self.class_probabilities_ = self.class_values_count_/100

        self.feature_values_ = np.array([np.unique(column) for column in X.T])
        self.feature_values_count_ = np.array([feature.shape[0] for feature in self.feature_values_])
        self._compute_probabilities(X, y)
        self._compute_smoothness_correction()
        self.is_fitted=True

    def _compute_smoothness_correction(self):
        term_1 = [np.log(count) for count in self.class_values_count_]
        term_3 = [-sum(np.log(self.class_values_count_[c] + self.alpha*self.feature_values_count_[j])
                       for j in range(self.column_count_))
                       for c in self.class_values_]
        self.smoothness_correction = np.array(term_1) + np.array(term_3)

    def _compute_probabilities(self, X: np.ndarray, y: np.ndarray):
        self.probabilities_ = np.array(_get_probabilities(
            X, y, self.feature_values_count_, self.n_classes_, self.alpha))

    def predict(self, X: np.ndarray):
        if not self.is_fitted:
            raise Exception("NB not fitted")
        if self.encode_data:
            X = self.feature_encoder_.transform(X)
        probabilities = _predict(X, self.probabilities_,self.feature_values_count_,self.alpha)
        probabilities += self.smoothness_correction
        output = np.argmax(probabilities, axis=1)
        if self.encode_data:
            output = self.class_encoder_.inverse_transform(output)
        return output

    def score(self, X: np.ndarray, y: np.ndarray):
        return np.sum(self.predict(X) == y)/y.shape[0]