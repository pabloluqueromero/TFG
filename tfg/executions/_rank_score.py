import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.ranker import RankerLogicalFeatureConstructor
from tfg.utils import get_X_y_from_database


def ranker_score_comparison(datasets, seed, test_size, base_path, params, n_iterations=30,n_intervals=5,metric="accuracy",send_email=False,email_data = dict()):
    result = []
    dataset_tqdm = tqdm(datasets)

    # Instantiate ranker
    r = RankerLogicalFeatureConstructor(n_intervals = n_intervals, metric=metric)
    nb = NaiveBayes(encode_data=True,n_intervals = n_intervals, metric=metric)
    for database in dataset_tqdm:
        name, label = database
        if os.path.exists(base_path+name):
            test = f"{name}.test.csv"
            data = f"{name}.data.csv"
            X, y = get_X_y_from_database(base_path, name, data, test, label)

            dataset_tqdm.set_postfix({"DATABASE": name})

            seed_tqdm = tqdm(range(n_iterations), leave=False)

            # Set up data structures to store results
            nb_score = np.zeros(shape=(len(params), n_iterations))
            r_score = np.zeros(shape=(len(params), n_iterations))
            r_combinations = np.zeros(shape=(len(params), n_iterations))
            r_selected = np.zeros(shape=(len(params), n_iterations))
            r_dummy = np.zeros(shape=(len(params), n_iterations))
            r_total_constructed = np.zeros(shape=(len(params), n_iterations))
            r_total_selected = np.zeros(shape=(len(params), n_iterations))
            r_original_selected = np.zeros(shape=(len(params), n_iterations))

            for i in seed_tqdm:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=seed+i,
                        stratify=y,
                        shuffle=True)
                except:
                    #Not enough values to stratify y
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=seed+i,
                        shuffle=True)

                nb.fit(X=X_train, y=y_train)
                naive_bayes_score = nb.score(X_test, y_test)
                c = CustomOrdinalFeatureEncoder(n_intervals = n_intervals)
                X_train = c.fit_transform(X_train)
                X_test = c.transform(X_test)
                l = CustomLabelEncoder()
                y_train = l.fit_transform(y_train)
                y_test = l.transform(y_test)

                conf_index = 0
                for conf in params:
                    seed_tqdm.set_postfix({"seed": i, "config": conf_index})
                    r.set_params(**conf)
                    # Fit
                    if conf_index == 0:
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

                    conf_index += 1

            # Insert to final result averaged metrics for this database
            for conf_index, conf in enumerate(params):
                row = [name,
                       X.shape[1],
                       np.mean(nb_score[conf_index]),
                       np.mean(r_score[conf_index]),
                       conf,
                       np.mean(r_combinations[conf_index]),
                       np.mean(r_selected[conf_index]),
                       np.mean(r_dummy[conf_index])]
                result.append(row)

        else:
            print(f"{name} doesnt' exist")
    columns = ["Database", "Number of attributes", "NBScore", "Ranker Score",
               "Configuration", "Combinations", "Selected_attributes", "Original"]
    result = pd.DataFrame(result, columns=columns)
    if send_email:
        from tfg.utils import send_results
        send_results("RANKER",email_data,result)
    return result
