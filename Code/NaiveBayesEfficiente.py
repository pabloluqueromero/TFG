import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.naive_bayes import GaussianNB,BernoulliNB,CategoricalNB
from NaiveBayesL import NaiveBayesL as NBP
from time import time
import random
from numba import jit

@jit(nopython=False)
def _get_probabilities(X, y, len_feature_values, n_classes, alpha):
    probabilities = []
    for i in range(X.shape[1]):
        counts = _get_counts(X[:, i], y, len_feature_values[i], n_classes)
        probabilities.append(np.log(counts+alpha))
    return probabilities

@jit(nopython=False)
def _get_counts(column, y, len_features, n_classes):
    counts = np.zeros((len_features, n_classes))
    for i in range(column.shape[0]):
        counts[column[i],y[i]]+=1
    return counts


# @jit
def _feature_encode(X, encoding):
    X = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = encoding[j][X[i, j]]
    return X


# @jit
def _class_encode(y, encoding):
    y = y.copy()
    for i in range(y.shape[0]):
        y[i] = encoding[y[i]]
    return y


def _class_decode(y, decoding):
    y = y.copy()
    for i in range(y.shape[0]):
        y[i] = decoding[y[i]]
    return y

# @jit
def _predict(X, probabilities):
    log_probability = np.zeros((X.shape[0], probabilities[0].shape[0]))
    for i in range(X.shape[1]):
        log_probability += probabilities[i][X[:,i],:]
    return log_probability


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, attribute_labels=None, alpha=1):
        self.attribute_labels = attribute_labels
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.row_count_, self.column_count_ = X.shape
        self.class_decoder_ = dict(enumerate(np.unique(y)))
        self.class_encoding_ = {v: k for k, v in self.class_decoder_.items()}
        self.feature_encoding_ = [dict(zip(unique_values,range(len(unique_values)))) for unique_values in np.array([np.unique(column) for column in X.T])]
        X = _feature_encode(X, self.feature_encoding_)
        y = _class_encode(y, self.class_encoding_)

        self.class_values_ = np.unique(y)
        self.class_values_count_ = np.bincount(y)
        self.n_classes_ = self.class_values_.shape[0]
        self.class_probabilities_ = self.class_values_count_/100
        self.feature_values_ = np.array([np.unique(column) for column in X.T])
        self.feature_values_count_ = np.array([feature.shape[0] for feature in self.feature_values_])
        self._compute_probabilities(X, y)
        self._compute_terms_1_and_3()

    def _compute_terms_1_and_3(self):
            t1 = [np.log(count) for count in self.class_values_count_]
            t3 = [-sum(np.log(self.class_values_count_[c] + self.alpha*self.feature_values_count_[f]) 
                            for f in range(self.column_count_)) 
                            for c in self.class_values_]
            self.terms_1_and_3 = np.array(t1) + np.array(t3)
    def _compute_probabilities(self, X, y):
        self.probabilities_=np.array(_get_probabilities(X, y, self.feature_values_count_, self.n_classes_, self.alpha))

    
    def predict(self, X):
        check_is_fitted(self)
        X=_feature_encode(X, self.feature_encoding_)
        probabilities = _predict(X,self.probabilities_)
        probabilities += self.terms_1_and_3
        return _class_decode(np.argmax(probabilities, axis=1),self.class_decoder_)
        # return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y=_class_encode(y, self.class_encoding_)
        return np.sum(self.predict(X) == y)/y.shape[0]


if __name__ == "__main__":
    attributes=[
        'X1',
        'X2'
    ]
    class_to_predict='C'

    print("------------Generating random Database-----------------")
    database_size=1000
    seed=650
    print(f"Database size: {database_size}")
    # Prepare data
    random.seed(seed)
    # data = np.array([ [random.randint(0,15),random.randint(1,20),random.randint(0,1)]  for _ in range(database_size) ])
    np.random.seed(seed)

    def generate_data(n, m, rx, ry=3):
        X = np.random.randint(rx, size=(n, m))
        y =  np.random.randint(ry, size=(n))
        return X, y

    N = 10000
    M = 4

    nbp = NBP(attributes=list(map(str,range(M))), class_to_predict="C")
    X, y = generate_data(N, M, 3, 3)
    print("Training classifiers\n")
    nb_classifier=NaiveBayes()
    gnb=GaussianNB()
    bnb=BernoulliNB()
    cnb=CategoricalNB()

    ts=time()
    gnb.fit(X, y)
    print(f"GaussianNB {gnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    bnb.fit(X, y)
    print(f"BernouilliNB {bnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    cnb.fit(X, y)
    print(f"CategoricalNB {cnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    nb_classifier.fit(X,y)
    print(f"CustomNB {nb_classifier.score(X,y)}  -> {time()-ts}")
    ts=time()
    nbp.fit(X,y)
    print(f"CustomNB2 {nbp.score(X,y)}  -> {time()-ts}")

    print("\nIteration2")
    ts=time()
    gnb.fit(X, y)
    print(f"GaussianNB {gnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    bnb.fit(X, y)
    print(f"BernouilliNB {bnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    cnb.fit(X, y)
    print(f"CategoricalNB {cnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    nb_classifier.fit(X,y)
    print(f"CustomNB {nb_classifier.score(X,y)}  -> {time()-ts}")
    ts=time()
    nbp.fit(X,y)
    print(f"CustomNB2 {nbp.score(X,y)}  -> {time()-ts}")
    print("\nIteration3")
    ts=time()
    gnb.fit(X, y)
    print(f"GaussianNB {gnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    bnb.fit(X, y)
    print(f"BernouilliNB {bnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    cnb.fit(X, y)
    print(f"CategoricalNB {cnb.score(X,y)}  -> {time()-ts}")
    ts=time()
    nb_classifier.fit(X,y)
    print(f"CustomNB {nb_classifier.score(X,y)}  -> {time()-ts}")
    ts=time()
    nbp.fit(X,y)
    print(f"CustomNB2 {nbp.score(X,y)}  -> {time()-ts}")
