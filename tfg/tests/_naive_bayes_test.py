import numpy as np
import random
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from time import time

from tfg.naive_bayes import PandasNaiveBayes
from tfg.naive_bayes import NaiveBayes as CustomNaiveBayes
from tfg.encoder import CustomOrdinalFeatureEncoder


'''
Method to test NB
'''


def test_remove_feature_with_index():

    X, y = make_classification(n_samples=1000,
                               n_features=100,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=1,
                               weights=None,
                               class_sep=1.0,
                               hypercube=True,
                               scale=2.0,
                               shuffle=True,
                               random_state=0)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X, y)
    nb.remove_feature(0)
    independent = nb.indepent_term_
    smoothed_log_counts_ = nb.smoothed_log_counts_
    removed = nb.predict_proba(np.delete(X, 0, axis=1))
    nb.fit(np.delete(X, 0, axis=1), y)
    og = nb.predict_proba(np.delete(X, 0, axis=1))
    assert np.allclose(nb.smoothed_log_counts_, smoothed_log_counts_)
    assert np.allclose(nb.indepent_term_, independent)
    assert np.allclose(og, removed)


def test_remove_feature():

    X, y = make_classification(n_samples=1000,
                               n_features=100,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=1,
                               weights=None,
                               class_sep=1.0,
                               hypercube=True,
                               scale=2.0,
                               shuffle=True,
                               random_state=0)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X, y)
    nb.remove_feature(0)
    independent = nb.indepent_term_
    smoothed_log_counts_ = nb.smoothed_log_counts_
    removed = nb.predict_proba(np.delete(X, 0, axis=1))
    nb.fit(np.delete(X, 0, axis=1), y)
    og = nb.predict_proba(np.delete(X, 0, axis=1))
    assert np.allclose(nb.smoothed_log_counts_, smoothed_log_counts_)
    assert np.allclose(nb.indepent_term_, independent)
    assert np.allclose(og, removed)


def test_add_features_with_index():
    X, y = make_classification(n_samples=1000,
                               n_features=100,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=1,
                               weights=None,
                               class_sep=1.0,
                               hypercube=True,
                               scale=2.0,
                               shuffle=True,
                               random_state=0)
    X_og = X.copy()
    index = [0, 8, 9, 20]
    X_two_less = np.delete(X_og, index, axis=1)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X_two_less, y)
    nb.add_features(X_og[:, index], y, index=index)
    independent = nb.indepent_term_
    smoothed_log_counts_ = nb.smoothed_log_counts_
    added = nb.predict_proba(X)

    nb.fit(X, y)
    og = nb.predict_proba(X)
    assert np.allclose(nb.indepent_term_, independent)
    assert np.allclose(nb.smoothed_log_counts_, smoothed_log_counts_)
    assert np.allclose(og, added)


def test_add_features():
    X, y = make_classification(n_samples=1000,
                               n_features=100,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=1,
                               weights=None,
                               class_sep=1.0,
                               hypercube=True,
                               scale=2.0,
                               shuffle=True,
                               random_state=0)
    X_og = X.copy()
    elments = [0, 2, 9, 48, 10]
    X_two_less = np.delete(X, elments, axis=1)
    X = np.concatenate([X_two_less, X[:, elments]], axis=1)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X_two_less, y)
    nb.add_features(X_og[:, elments], y)
    independent = nb.indepent_term_
    smoothed_log_counts_ = nb.smoothed_log_counts_
    added = nb.predict_proba(X)

    nb.fit(X, y)
    og = nb.predict_proba(X)
    assert np.allclose(nb.indepent_term_, independent)
    assert np.allclose(nb.smoothed_log_counts_, smoothed_log_counts_)
    assert np.allclose(og, added)
