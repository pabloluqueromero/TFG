import pandas as pd


from tfg.executions import *
import subprocess
import argparse
from time import sleep
import numpy as np
import itertools
from tfg.utils._utils import get_X_y_from_database
from tqdm.std import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.naive_bayes import NaiveBayes
from tfg.optimization.genetic_programming import GeneticProgrammingRankMutation
from tfg.optimization.ant_colony import ACFCS
from tfg.ranker import RankerLogicalFeatureConstructor

'''
Global Variables
'''
seed = 200
n_repeats = 5
n_splits = 3
n_intervals = 5
base_path = "./UCIREPO/"
datasets = [
    ["lenses", "ContactLens"],
    ["abalone", "Rings"],
    ["anneal", "label"],
    ["audiology", "label"],
    ["balance-scale", "label"],
    ["breast-cancer", "Class"],
    ["car-evaluation", "safety"],
    ["cmc", "Contraceptive"],
    ["credit", "A16"],
    ["cylinder-bands", "band type"],
    # ["hill_valley", "class"],
    ["derm", "class"],
    ["electricgrid", "stabf"],
    ["glass", "Type"],
    ["horse-colic", "surgery"],
    ["iris", "Species"],
    ["krkp", "label"],
    ["mammographicmasses", "Label"],
    ["mushroom", "class"],
    ["pima", "Outcome"],
    ["student", "Walc"],
    ["voting", "Class Name"],
    ["wine", "class"],
    ["wisconsin", "diagnosis"],
    ["yeast", "nuc"],
    ["tictactoe", "class"],
    ["spam", "class"]
]

label = {dataset: label for dataset, label in datasets}
metric = "accuracy"
df = pd.read_csv("configs.csv", sep=";")
configs = dict()
import ast
from numpy import nan
for index in range(df.shape[0]):
    algorithm, database, config = df.iloc[index,:]
    if algorithm not in configs:
        configs[algorithm] = dict()
    a = eval(config,locals())
    configs[algorithm][database] = a


n_splits = 3
n_repeats = 5
seed = 200

result = [["Database","Algorithm","Scores"]]
for algorithm in tqdm(configs.keys()):
    for name,config in tqdm(configs[algorithm].items()):

        test = f"{name}.test.csv"
        data = f"{name}.data.csv"
        X, y = get_X_y_from_database(base_path, name, data, test, label[name])


        # Create splits for the experiments
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        seed_tqdm = tqdm(rskf.split(X, y))
        nb = NaiveBayes(encode_data=False)
        # Execute experiments
        scores = []
        nb_scores = []
        for data in rskf.split(X, y):
            train_index, test_index = data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Encode the data
            c = CustomOrdinalFeatureEncoder(n_intervals=n_intervals)
            X_train = c.fit_transform(X_train)
            X_test = c.transform(X_test)
            l = CustomLabelEncoder()
            y_train = l.fit_transform(y_train)
            y_test = l.transform(y_test)

            # Assess the classifiers reusing info to speed up evaluation
            nb.fit(X_train, y_train)
            nb_scores.append(nb.score(X_test, y_test))

            if "genetic" in algorithm:
                clf = GeneticProgrammingRankMutation()
                clf.reset_evaluation()
                algo_name = "GENETIC"
            elif "aco" in algorithm:
                clf = ACFCS()
                clf.reset_cache()
                algo_name = "ACO"
            else:
                clf = RankerLogicalFeatureConstructor()
                algo_name = "RANKER"


            # Reset evaluation-cache for new split
            if "encode" in config:
                del config["encode"]
                config["encode_data"] = True
            if "backwards" in config:
                del config["backwards"]
            clf.set_params(**config)
            clf.fit(X_train, y_train)
            # score
            scores.append(clf.score(X_test, y_test))
        result.append([name,algo_name,scores])
        result.append([name,"NB",nb_scores])
    pd.DataFrame(result).to_csv("temp.csv",index=False)
pd.DataFrame(result).to_csv("temp.csv",index=False)
