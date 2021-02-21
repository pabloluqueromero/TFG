import numpy as np
import pandas as pd

from itertools import product
from sklearn.datasets import make_classification
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from tqdm.autonotebook  import tqdm

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.naive_bayes import NaiveBayes
from tfg.utils import make_discrete
from time import time




def evaluate(X_train, y_train, X_test, y_test, clf, fit_time, predict_time, score):
    ts = time()
    clf.fit(X_train, y_train)
    fit_time.append(time()-ts)

    ts = time()
    s = clf.score(X_test, y_test)
    predict_time.append(time()-ts)
    score.append(s)
    return 0


def update_df(df,
              clf,
              rows,
              columns,
              nb_fit_time,
              nb_predict_time,
              nb_score,
              nb_errors,
              filename):
    row = [clf,
           rows,
           columns,
           np.mean(nb_fit_time),
           np.std(nb_fit_time),
           0 if nb_errors else np.mean(nb_predict_time),
           0 if nb_errors else np.std(nb_predict_time),
           0 if nb_errors else np.mean(nb_score),
           nb_errors]
    df = df.append([row])
    if filename:
        df.to_csv(filename)


def timing_comparison(combinations=None, n_iterations=15, verbose=1, seed=200, filename=None):
    column_names = ["Classifier",
                    "n_sampples",
                    "n_features",
                    "average_fit_time",
                    "std_fit_time",
                    "average_predict_time",
                    "std_predict_time",
                    "score"]

    df = pd.DataFrame(columns=column_names)
    if combinations is None:
        columns = range(10, 40010, 5000)
        rows = [10, 100, 1000]
        combinations = list(product(rows, columns)) + \
            list(product(columns, rows))
        combinations += [(500000, 100), (500000, 10)]

    clf_no_encoding = NaiveBayes(encode_data=False, alpha=1)
    clf_encoding = NaiveBayes(encode_data=True, alpha=1)
    clf_categorical_sklearn = CategoricalNB(alpha=1)
    clf_gaussian_sklearn = GaussianNB()
    progress_bar = tqdm(total=len(combinations), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    for n_features, n_samples in combinations:
        if verbose:
            progress_bar.set_postfix({"n_samples":n_samples, "n_features":n_features})
            progress_bar.update(1)
            progress_bar.refresh()
        X, y = make_classification(n_samples=n_samples,
                                   n_features=n_features,
                                   n_informative=n_features-1,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_classes=2,
                                   n_clusters_per_class=2,
                                   weights=None,
                                   flip_y=0.01,
                                   class_sep=1.0,
                                   hypercube=True,
                                   shift=0.0,
                                   scale=2.0,
                                   shuffle=True,
                                   random_state=seed)
        X = make_discrete(X, m=1)

        X_train, X_test, y_train, y_test = X, X, y, y
        gaussian_nb_fit_time = []
        gaussian_nb_predict_time = []
        gaussian_nb_score = []
        gaussian_nb_errors = 0

        categorical_nb_fit_time = []
        categorical_nb_predict_time = []
        categorical_nb_score = []
        categorical_nb_errors = 0

        custom_no_encoding_nb_fit_time = []
        custom_no_encoding_nb_predict_time = []
        custom_no_encoding_nb_score = []
        custom_no_encoding_nb_errors = 0

        custom_encoding_nb_fit_time = []
        custom_encoding_nb_predict_time = []
        custom_encoding_nb_score = []
        custom_encoding_nb_errors = 0

        for _ in range(n_iterations):
            gaussian_nb_errors += evaluate(X_train, y_train, X_test, y_test,
                                           clf_gaussian_sklearn,
                                           gaussian_nb_fit_time,
                                           gaussian_nb_predict_time,
                                           gaussian_nb_score)
            categorical_nb_errors += evaluate(X_train, y_train, X_test, y_test,
                                              clf_categorical_sklearn,
                                              categorical_nb_fit_time,
                                              categorical_nb_predict_time,
                                              categorical_nb_score)
            custom_no_encoding_nb_errors += evaluate(X_train, y_train, X_test, y_test,
                                                     clf_no_encoding,
                                                     custom_no_encoding_nb_fit_time,
                                                     custom_no_encoding_nb_predict_time,
                                                     custom_no_encoding_nb_score)
            custom_encoding_nb_errors += evaluate(X_train, y_train, X_test, y_test,
                                                  clf_encoding,
                                                  custom_encoding_nb_fit_time,
                                                  custom_encoding_nb_predict_time,
                                                  custom_encoding_nb_score)

        update_df(df,
                  "Gaussian",
                  n_samples,
                  n_features,
                  gaussian_nb_fit_time,
                  gaussian_nb_predict_time,
                  gaussian_nb_score,
                  gaussian_nb_errors,
                  filename)
        update_df(df,
                  "Categorical",
                  n_samples,
                  n_features,
                  categorical_nb_fit_time,
                  categorical_nb_predict_time,
                  categorical_nb_score,
                  categorical_nb_errors,
                  filename)

        update_df(df,
                  "Custom_nb_encoding",
                  n_samples,
                  n_features,
                  custom_encoding_nb_fit_time,
                  custom_encoding_nb_predict_time,
                  custom_encoding_nb_score,
                  custom_encoding_nb_errors,
                  filename)

        update_df(df,
                  "Custom_nb_no_encoding",
                  n_samples,
                  n_features,
                  custom_no_encoding_nb_fit_time,
                  custom_no_encoding_nb_predict_time,
                  custom_no_encoding_nb_score,
                  custom_no_encoding_nb_errors,
                  filename)
    return df
