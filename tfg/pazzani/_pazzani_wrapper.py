import numpy as np
import pandas as pd

from itertools import combinations
from itertools import product
from collections import deque
from sklearn.model_selection import LeaveOneOut
from sklearn.base import BaseEstimator
# Local imports
from tfg.naive_bayes import NaiveBayes
from tfg.utils import join_columns, concat_columns, flatten, memoize, combine_columns


from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder


class PazzaniWrapper(BaseEstimator):
    """Generic Pazzani Wrapper .

    Feature subset selection and constrution. Algorithms implemented:
        - Backward Sequential Elimination and Joining (BSEJ)
        - Forward Sequential Selection and Joining (FSSJ).

    Parameters
    ----------

    seed : float or None
        Seed used for the generation of random values for the pheromone trail.

    cv : instance of an object extending BaseCrossValidator
        Cross validation used to assess the goodness of a particular subset

    strategy : {'BSEJ','FSSJ'}
        Strategy for the subset selection and feature construction

    verbose :int {0,1}, default = 1 
        Display process progress

    Attributes
    ----------
    features_ : array-like of Feature
        Selected features after fit process

    feature_transformer_ : function
        Lambda function to transform prediction variables

    feature_encoder_  : CustomOrdinalFeatureEncoder
        OrdinalEncoder for the predictive variables

    class_encoder_  : CustomLabelEncoder
        LabelEncoder for the target class

    _fit : function object
        Function selected according to the strategy

    evaluate : function object
        Memoized function for evaluation

    """

    def __init__(self,
                 seed=None,
                 cv=None,
                 strategy="BSEJ",
                 encode_data=True,
                 verbose=0):
        self.cv = cv
        self.classifier = NaiveBayes(encode_data=True)
        self.seed = seed
        self.strategy = strategy
        self.verbose = verbose
        self.encode_data = encode_data
        self._fit = self.fit_bsej if self.strategy == "BSEJ" else self.fit_fssj

        allowed_strategies = ("BSEJ", "FSSJ")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown strategy type: %s, expected one of %s." % (self.strategy, allowed_strategies))

    def fit(self, X, y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()
        X = self.feature_encoder_.fit_transform(X).astype(str)
        y = self.class_encoder_.fit_transform(y)
        return self._fit(X, y)

    def _generate_neighbors_bsej(self, current_columns, X):
        '''
            Neighbor generation based on  (current):
                - Remove feature from current
                - Join features a and b, and remove a and b
        '''
        if X.shape[1] > 1:
            for col in range(X.shape[1]):
                new_columns = current_columns.copy()
                del new_columns[col]
                yield new_columns, np.delete(X, col, axis=1)
            for features in combinations(np.arange(X.shape[1]), 2):
                new_col_name = flatten([current_columns[features[0]], current_columns[features[1]]])
                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                features = sorted(features, reverse=True)
                del new_columns[features[0]]
                del new_columns[features[1]]

                columns = features
                combined_columns = combine_columns(X, columns)
                neighbor = np.concatenate([X, combined_columns], axis=1)
                yield new_columns, np.delete(neighbor, columns, axis=1)

    def fit_bsej(self, X, y):
        self.evaluate = memoize(_evaluate, attribute_to_cache="columns")
        current_best = X.copy()
        current_columns = deque(range(X.shape[1]))
        best_score = self.evaluate(self.classifier, self.cv, current_best, y, columns=current_columns)
        stop = False
        while not stop:
            stop = True
            if self.verbose:
                print("Current Best: ", current_columns, " Score: ", best_score)
            for new_columns, neighbor in self._generate_neighbors_bsej(current_columns, current_best):
                score = self.evaluate(self.classifier, self.cv, neighbor, y, columns=new_columns)
                if self.verbose == 2:
                    print("\tNeighbor: ", new_columns, " Score: ", score)
                if score > best_score:
                    stop = False
                    current_best = neighbor
                    best_score = score
                    current_columns = new_columns
                    if score == 1.0:
                        stop = True
                        break

        if self.verbose:
            print("Final best: ", list(current_columns), " Score: ", best_score)
        self.features_ = current_columns
        self.feature_transformer = lambda X: join_columns(X, columns=self.features_)
        model = self.classifier.fit(self.feature_transformer(X), y)
        return self

    def _generate_neighbors_fssj(self, current_columns, individual, original_data, available_columns):
        '''
            Neighbor generation based on  (current,available):
                - Add feature from available to current
                - Join one feature from available to with one feature from current
        '''
        if available_columns:
            for index, col in enumerate(available_columns):
                new_columns = current_columns.copy()
                new_columns.append(col)
                new_available_columns = available_columns.copy()
                del new_available_columns[index]
                if individual is not None:
                    neighbor = np.concatenate([individual, original_data[:, col].reshape(-1, 1)], axis=1)
                else:
                    neighbor = original_data[:, col].reshape(-1, 1)
                yield new_columns, new_available_columns, neighbor
        if individual is not None and individual.shape[1] > 0 and available_columns:
            for features_index in product(np.arange(len(available_columns)), np.arange(len(current_columns))):
                features = available_columns[features_index[0]], current_columns[features_index[1]]
                new_col_name = flatten([features[0], features[1]])

                new_available_columns = available_columns.copy()
                del new_available_columns[features_index[0]]

                new_columns = current_columns.copy()
                new_columns.append(tuple(new_col_name))
                del new_columns[features_index[1]]

                if isinstance(features[1], tuple):
                    features = list(features)
                    features[1] = list(features[1])
                    features = tuple(features)
                separated_columns = np.concatenate(
                    [original_data[:, features[0]].reshape(-1, 1), individual[:, features_index[1]].reshape(-1, 1)], axis=1)
                combined_columns = combine_columns(separated_columns)
                neighbor = np.concatenate([individual, combined_columns], axis=1)
                yield new_columns, new_available_columns, np.delete(neighbor, features_index[1], axis=1)

    def fit_fssj(self, X, y):
        self.evaluate = memoize(_evaluate, attribute_to_cache="columns")
        current_best = None
        current_columns = deque()
        available_columns = list(range(X.shape[1]))
        best_score = -float("inf")
        stop = False
        while not stop:
            stop = True
            if self.verbose:
                print("Current Best: ", current_columns, " Score: ",
                      best_score, "Available columns: ", available_columns)
            for new_columns, new_available_columns, neighbor in self._generate_neighbors_fssj(current_columns=current_columns,
                                                                                              individual=current_best,
                                                                                              original_data=X,
                                                                                              available_columns=available_columns):
                score = self.evaluate(self.classifier, self.cv, neighbor, y, columns=new_columns)
                if self.verbose == 2:
                    print("\tNeighbour: ", new_columns, " Score: ", score, "Available columns: ", new_available_columns)
                if score > best_score:
                    stop = False
                    current_best = neighbor
                    best_score = score
                    current_columns = new_columns
                    available_columns = new_available_columns
                    if score == 1.0:
                        stop = True
                        break
        if self.verbose:
            print("Final best: ", list(current_columns), " Score: ", best_score)
        self.features_ = current_columns
        self.feature_transformer = lambda X: join_columns(X, columns=self.features_)
        model = self.classifier.fit(self.feature_transformer(X), y)
        return self

    def evaluate(self, classifier, cv, X, y, columns):
        return _evaluate(classifier, cv, X, y, columns)

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = self.feature_encoder_.transform(X).astype(str)
        if y is not None:
            y = self.class_encoder_.transform(y)
        return self.feature_transformer(X), y

    def predict(self, X):
        X, _ = self.transform(X)
        if self.encode_data:
            return self.class_encoder_.inverse_transform(self.classifier.predict(X))
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        X, _ = self.transform(X)
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        X, y = self.transform(X, y)
        return self.classifier.score(X, y)


def _evaluate(clf, cv, X, y, columns):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    scores = []
    if cv:
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        return np.mean(scores)
    return clf.leave_one_out_cross_val(X, y)
