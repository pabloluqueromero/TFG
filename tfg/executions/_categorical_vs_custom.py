import os
import numpy as np
import pandas as pd

from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.ranker import RankerLogicalFeatureConstructor
from tfg.utils import get_X_y_from_database


def compare_categorical_custom_score(databases, seed, test_size, base_path,n_iterations = 30):
    df_result_columns = ["CategoricalNB","CustomNB"]
    for database in databases:
        result = [] 
        database_tqdm = tqdm(databases)

        # Instantiate ranker
        cb = CategoricalNB()
        nb = NaiveBayes(encode_data=True)
        for database in database_tqdm:
            name, label = database
            if os.path.exists(base_path+name):
                test = f"{name}.test.csv"
                data = f"{name}.data.csv"
                X, y = get_X_y_from_database(base_path, name, data, test, label)

                database_tqdm.set_postfix({"DATABASE": name})

                seed_tqdm = tqdm(range(n_iterations), leave=False)




