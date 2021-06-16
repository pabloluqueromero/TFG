import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.ranker import RankerLogicalFeatureConstructor
from tfg.utils import get_X_y_from_database

"""
    Method to run experiments on the GeneticProgramming algorithm against the Naive Bayes classifier.

    Ordinal encoder using numpy's searchsorted method to encode data to an integer ordinal representation.
    Automatic handling of unseen values (transformed to n, where n is the number of unique values for a feature)

    Discretizes in 5 intervales using quantile strategy, numerical features are detected only 
    when the provided features are wrapped in a pandas DataFrame and have dtype float.

    Parameters
    ----------

    datasets : array-like of tuples
        List of tuples where each elements is of the type: ('dataset_name','target_name')

    base_path: str
        Path to the folder with all the datasets

    params : array-like 
        List containing dictionaries with the different combinations of hyper-parameters

    seed : int or None
        Seed to guarantee reproducibility
        
    n_splits : int, default=3
        Number of folds for the Repeated Stratified K-fold cross-validation
        
    n_repeats : int, default=5
        Number of iterations for the Repeated Stratified K-fold cross-validation
    
    n_intervals : int, default=5
        Number of intervals for the iscretization inside CustomOrdinalFeatureEncoder

    metric : str, default="accuracy"
        Target metric for the algorithm to optimise

    send_email : bool, default=False
        Send the resulting csv by email. Requires the email_data to be filled

    email_data : dict
        Dictionary containing the data for sending the email
        Fields
            - FROM : sender/receiver
            - TO : receiver
            - PASSWORD : password
            - TITLE : Subject
            - FILENAME : filename for the file to be sent
"""


def ranker_score_comparison(datasets,
                            seed,
                            base_path,
                            params,
                            n_splits=3,
                            n_repeats=5,
                            n_intervals=5,
                            metric="accuracy",
                            send_email=False,
                            email_data=dict(),
                            share_rank=True):
    result = []
    columns = ["Database",
               "Number of attributes",
               "NBScore",
               "NBScore STD",
               "Ranker Score",
               "Ranker Score STD",
               "Configuration",
               "Combinations",
               "Selected_attributes",
               "Original"]

    dataset_tqdm = tqdm(datasets)

    # Instantiate the classifier
    r = RankerLogicalFeatureConstructor(n_intervals=n_intervals, metric=metric)
    nb = NaiveBayes(encode_data=False, n_intervals=n_intervals, metric=metric)

    # Execute algorithm on datasets
    for database in dataset_tqdm:
        name, label = database
        if not os.path.exists(base_path + name):
            print(f"{name} doesnt' exist")
            continue
        # Assume UCI REPO like data
        test = f"{name}.test.csv"
        data = f"{name}.data.csv"
        X, y = get_X_y_from_database(base_path, name, data, test, label)

        dataset_tqdm.set_postfix({"DATABASE": name})

        # Set up data structures to store results
        nb_score = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_score = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_combinations = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_selected = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_dummy = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_total_constructed = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_total_selected = np.zeros(shape=(len(params), n_splits*n_repeats))
        r_original_selected = np.zeros(shape=(len(params), n_splits*n_repeats))

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        seed_tqdm = tqdm(rskf.split(X, y),
                         leave=False,
                         total=n_splits*n_repeats,
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

        for i, data in enumerate(seed_tqdm):
            train_index, test_index = data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            c = CustomOrdinalFeatureEncoder(n_intervals=n_intervals)
            X_train = c.fit_transform(X_train)
            X_test = c.transform(X_test)
            l = CustomLabelEncoder()
            y_train = l.fit_transform(y_train)
            y_test = l.transform(y_test)

            # Assess the classifiers
            nb.fit(X=X_train, y=y_train)
            naive_bayes_score = nb.score(X_test, y_test)

            for conf_index, conf in enumerate(params):
                seed_tqdm.set_postfix({"config": conf_index})
                r.set_params(**conf)
                # Fit
                if conf_index == 0 or not share_rank:
                    # The rank is computed from scratch
                    r.fit(X_train, y_train)
                else:
                    r.filter_features(r.feature_encoder_.transform(
                        X_train), r.class_encoder_.transform(y_train))

                # score
                ranker_score = r.score(X_test, y_test)

                # Get data
                n_original_features = len(list(filter(lambda x: isinstance(
                    x, DummyFeatureConstructor), r.final_feature_constructors)))
                n_combinations = len(r.all_feature_constructors)
                n_selected = len(r.final_feature_constructors)

                # Update
                nb_score[conf_index, i] = naive_bayes_score
                r_score[conf_index, i] = ranker_score
                r_combinations[conf_index, i] = n_combinations
                r_selected[conf_index, i] = n_selected
                r_dummy[conf_index, i] = n_original_features

        # Insert to final result averaged metrics for this dataset
        for conf_index, conf in enumerate(params):
            row = [name,
                   X.shape[1],
                   np.mean(nb_score[conf_index]),
                   np.std(nb_score[conf_index]),
                   np.mean(r_score[conf_index]),
                   np.std(r_score[conf_index]),
                   conf,
                   np.mean(r_combinations[conf_index]),
                   np.mean(r_selected[conf_index]),
                   np.mean(r_dummy[conf_index])]
            result.append(row)
    result = pd.DataFrame(result, columns=columns)
    if send_email:
        from tfg.utils import send_results
        send_results("RANKER", email_data, result)
    return result
