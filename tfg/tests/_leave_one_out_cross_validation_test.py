import numpy as np
import random

from functools import reduce
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from time import time

# Local imports:
from tfg.naive_bayes import NaiveBayes
from tfg.encoder import CustomOrdinalFeatureEncoder


def cross_leave_one_out(clf, X, y):
    l = LeaveOneOut()
    score_avg = []
    for train_index, test_index in l.split(X):
        clf.fit(X[train_index], y[train_index])
        score_avg.append(clf.predict(
            X[test_index])[0] == y[test_index])
    return np.mean(score_avg)


def test_incremental_validation(X=None, y=None, iterations=10,verbose=1):
    if not X:
        X, y = make_classification(n_samples=10000,
                                   n_features=100,
                                   n_informative=2,
                                   n_redundant=10,
                                   n_repeated=0,
                                   n_classes=2,
                                   n_clusters_per_class=2,
                                   weights=None,
                                   class_sep=1,
                                   hypercube=False,
                                   scale=1.0,
                                   shuffle=True,
                                   random_state=0)
        X//=10 #--> To be able to evaluate categoricalNB

    # classifiers
    nb_classifier = NaiveBayes(encode_data=True)
    nb_classifier_no_encoding = NaiveBayes(encode_data=False)
    custom_encoder = CustomOrdinalFeatureEncoder()
    cnb = CategoricalNB()

    # accumulators
    categorical_nb = []
    custom_nb_val_1 = []
    custom_nb_val_2 = []
    custom_nb_val_3 = []
    custom_nb_val_4 = []
    for i in range(iterations):
        if verbose:
            print(f"Iteration {i}")
        ts = time()
        X2 = custom_encoder.fit_transform(X)
        score_1 = cross_leave_one_out(cnb, X2, y)
        categorical_nb.append(time()-ts)

        ts = time()
        score_2 = nb_classifier.leave_one_out_cross_val(X, y)
        custom_nb_val_1.append(time()-ts)

        ts = time()
        score_3 = nb_classifier.leave_one_out_cross_val2(X, y)
        custom_nb_val_2.append(time()-ts)

        ts = time()
        score_4 = cross_leave_one_out(nb_classifier, X, y)
        custom_nb_val_3.append(time()-ts)

        ts = time()
        X2 = custom_encoder.fit_transform(X)
        score_5 = cross_leave_one_out(nb_classifier_no_encoding, X2, y)
        custom_nb_val_4.append(time()-ts)

        if i == 0 and verbose:
            assert reduce(lambda a,b: (a==b).all(), [score_1, score_2, score_3, score_4, score_5])

    print("Categorical with scikit loo: ", np.mean(categorical_nb[1:]))
    print("Custom with scikit loo: ", np.mean(custom_nb_val_3[1:]))
    print("Custom with scikit loo (pre-encoding): ", np.mean(custom_nb_val_4[1:]))
    print("Custom with first incremental: ", np.mean(custom_nb_val_1[1:]))
    print("Custom with second incremental: ", np.mean(custom_nb_val_2[1:]))

