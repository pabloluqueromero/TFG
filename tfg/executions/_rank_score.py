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


def ranker_score_comparison(datasets,
                            seed,
                            base_path,
                            params,
                            n_splits=3,
                            n_repeats=5,
                            n_intervals=5,
                            metric="accuracy",
                            send_email=False,
                            email_data = dict(),
                            share_rank=True):
    result = []
    dataset_tqdm = tqdm(datasets)

    # Instantiate ranker
    r = RankerLogicalFeatureConstructor(n_intervals = n_intervals, metric=metric)
    nb = NaiveBayes(encode_data=False,n_intervals = n_intervals, metric=metric)
    for database in dataset_tqdm:
        name, label = database
        if os.path.exists(base_path+name):
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

            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=seed)
            seed_tqdm = tqdm(rskf.split(X,y), 
                             leave=False,
                             total=n_splits*n_repeats, 
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
            i=-1
            for train_index, test_index  in seed_tqdm:
                i+=1
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                c = CustomOrdinalFeatureEncoder(n_intervals = n_intervals)
                X_train = c.fit_transform(X_train)
                X_test = c.transform(X_test)
                l = CustomLabelEncoder()
                y_train = l.fit_transform(y_train)
                y_test = l.transform(y_test)

                nb.fit(X=X_train, y=y_train)
                naive_bayes_score = nb.score(X_test, y_test)
                conf_index = 0
                for conf in params:
                    seed_tqdm.set_postfix({ "config": conf_index})
                    r.set_params(**conf)
                    # Fit
                    if conf_index == 0 or share_rank:
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
                       np.std(nb_score[conf_index]),
                       np.mean(r_score[conf_index]),
                       np.std(r_score[conf_index]),
                       conf,
                       np.mean(r_combinations[conf_index]),
                       np.mean(r_selected[conf_index]),
                       np.mean(r_dummy[conf_index])]
                result.append(row)

        else:
            print(f"{name} doesnt' exist")
    columns = ["Database", "Number of attributes", "NBScore","NBScore STD", "Ranker Score","Ranker Score STD",
               "Configuration", "Combinations", "Selected_attributes", "Original"]
    result = pd.DataFrame(result, columns=columns)
    if send_email:
        from tfg.utils import send_results
        send_results("RANKER",email_data,result)
    return result
