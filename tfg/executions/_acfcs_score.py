import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.ant_colony import ACFCS
from tfg.utils import get_X_y_from_database


def acfs_score_comparison(datasets, seed, test_size, base_path, params, n_iterations=30,n_intervals=5):
    result = []
    dataset_tqdm = tqdm(datasets)

    # Instantiate ranker
    acfcs = ACFCS(verbose=0)
    nb = NaiveBayes(encode_data=True,n_intervals=n_intervals)
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
            acfcs_score = np.zeros(shape=(len(params), n_iterations))
            acfcs_selection_matrix = np.zeros(shape=(len(params), n_iterations))
            acfcs_construction_matrix = np.zeros(shape=(len(params), n_iterations))
            acfcs_nodes = np.zeros(shape=(len(params), n_iterations))
            acfcs_dummy = np.zeros(shape=(len(params), n_iterations))
            acfcs_selected = np.zeros(shape=(len(params), n_iterations))

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

                nb.fit(X_train, y_train)
                naive_bayes_score = nb.score(X_test, y_test)
                print(naive_bayes_score)
                conf_index  = 0
                for conf in params:
                    seed_tqdm.set_postfix({"seed": i, "config": conf_index})
                    acfcs.set_params(**conf)
                    acfcs.fit(X_train, y_train)

                    # score
                    acfcs_score_conf = acfcs.score(X_test, y_test)

                    # Get data
                    n_original_features = len(list(filter(lambda x: isinstance(
                        x, DummyFeatureConstructor), acfcs.best_features)))
                    n_selected = len(acfcs.best_features)
                    selection_matrix = len(acfcs.afg.pheromone_matrix_selection)
                    construction_matrix = len(acfcs.afg.pheromone_matrix_attribute_completion)
                    nodes = len(acfcs.afg.nodes)
                    # Update
                    nb_score[conf_index, i] = naive_bayes_score
                    acfcs_score[conf_index, i] = acfcs_score_conf
                    acfcs_selection_matrix[conf_index, i] = selection_matrix
                    acfcs_construction_matrix[conf_index, i] = construction_matrix
                    acfcs_nodes[conf_index, i] = nodes
                    acfcs_dummy[conf_index, i] = n_original_features
                    acfcs_selected[conf_index, i] = n_selected

                    conf_index += 1

            # Insert to final result averaged metrics for this database
            for conf_index, conf in enumerate(params):
                row = [name,
                       X.shape[1],
                       np.mean(nb_score[conf_index]),
                       np.mean(acfcs_score[conf_index]),
                       conf,
                       np.mean(acfcs_nodes[conf_index]),
                       np.mean(acfcs_construction_matrix[conf_index]),
                       np.mean(acfcs_selection_matrix[conf_index]),
                       np.mean(acfcs_selected[conf_index]),
                       np.mean(acfcs_dummy[conf_index])]
                result.append(row)

        else:
            print(f"{name} doesnt' exist")
    columns = ["Database", 
               "Number of attributes", 
               "NBScore", 
               "ACFCS Score",
               "Configuration", 
               "Nodes",
               "Contruction Matrix",
               "Selection Matrix", 
               "Selected_attributes", 
               "Original"]
    result = pd.DataFrame(result, columns=columns)
    return result


# databases = [
#     # ["abalone", "Rings"],
#     # ["adult", "income"],
#     # ["anneal", "label"],
#     # ["audiology", "label"],
#     # ["balance-scale", "label"],
#     # ["krkopt", "Optimal depth-of-win for White"],
#     # ["iris", "Species"],
#     # ["horse-colic", "surgery"],
#     # ["glass", "Type"],
#     # ["krkp", "label"],
#     # ["mushroom", "class"],
#     # ["voting", "Class Name"],
#     # ["credit", "A16"],
#     # ["pima", "Outcome"],
#     # ["wine", "class"],
#     ["wisconsin", "diagnosis"]
# ]


# # # databases = [ [database[0]+"/"+database[0],database[1]]for database in databases]
# # base_path = "../input/dataset/"
# base_path = "./UCIREPO/"

# params = [
#     {
#     "ants":20, 
#     "evaporation_rate":0.1,
#     "intensification_factor":2,
#     "alpha":1, 
#     "beta":0.7, 
#     "beta_evaporation_rate":0.5,
#     "iterations":100, 
#     "early_stopping":3,
#     "seed": 3,
#     "parallel":False,
#     "save_features":False,
#     "verbose":0,
#     "graph_strategy":"mutual_info"
#     },
#     # {
#     # "ants":20, 
#     # "evaporation_rate":0.1,
#     # "intensification_factor":3,
#     # "alpha":0.5, 
#     # "beta":0.5, 
#     # "beta_evaporation_rate":0.5,
#     # "iterations":100, 
#     # "early_stopping":3,
#     # "seed": 3,
#     # "parallel":False,
#     # "save_features":False,
#     # "verbose":0,
#     # "graph_strategy":"mutual_info"
#     # }
# ]
# seed= 4
# test_size = 0.3
# df = acfs_score_comparison(datasets=databases,
#                       base_path=base_path,
#                       test_size=test_size,
#                       params=params,
#                       seed=seed)
# df.to_csv("result_acfcs.csv", index=False)

