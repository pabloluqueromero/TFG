



import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.genetic_algorithm import GeneticAlgorithm,GeneticAlgorithmV2
from tfg.utils import get_X_y_from_database


def genetic_score_comparison(datasets,
                            seed,
                            base_path,
                            params,
                            n_splits=3,
                            n_repeats=5,
                            n_intervals=5,
                            metric="accuracy",
                            send_email=False,
                            email_data = dict(),
                            version=1):
    result = []
    dataset_tqdm = tqdm(datasets)

    # Instantiate ranker
    r = GeneticAlgorithm(seed=200) if version==1 else GeneticAlgorithmV2(seed=200)
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
            r_selected = np.zeros(shape=(len(params), n_splits*n_repeats))
            r_dummy = np.zeros(shape=(len(params), n_splits*n_repeats))
            rskf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=seed)
            seed_tqdm = tqdm(rskf.split(X,y),
                             leave=False,
                             total=n_splits*n_repeats, 
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
            i=-1
            for train_index, test_index  in seed_tqdm:
                r.reset_evaluation()
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
                    r.fit(X_train, y_train)
                    
                    # score
                    ranker_score = r.score(X_test, y_test)

                    # Get data
                    n_original_features = len(list(filter(lambda x: isinstance(
                        x, DummyFeatureConstructor), r.best_features)))
                    n_selected = len(r.best_features)

                    # Update
                    nb_score[conf_index, i] = naive_bayes_score
                    r_score[conf_index, i] = ranker_score
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
                       np.mean(r_selected[conf_index]),
                       np.mean(r_dummy[conf_index])]
                result.append(row)

        else:
            print(f"{name} doesnt' exist")
    columns = ["Database", "Number of attributes", "NBScore","NBScore STD", "Genetic Score","Genetic Score STD",
               "Configuration", "Selected_attributes", "Original"]
    result = pd.DataFrame(result, columns=columns)
    if send_email:
        from tfg.utils import send_results
        send_results(f"GENETIC_{version}",email_data,result)
    return result
