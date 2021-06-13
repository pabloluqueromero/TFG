import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.optimization.ant_colony import ACFCS, AntFeatureGraphMI
from tfg.utils import get_X_y_from_database

def graph_sizes(datasets, 
                seed,
                base_path, 
                n_splits=3,
                n_repeats=5,
                n_intervals=5,):
    dataset_tqdm = tqdm(datasets)
    connections_list = [1,3,5,10]
    results  = []
    for database in dataset_tqdm:
        edges_selection = np.zeros((len(connections_list),n_intervals*n_repeats))
        edges_construction = np.zeros((len(connections_list),n_intervals*n_repeats))
        nodes = np.zeros(n_intervals*n_repeats)
        name, label = database
        if os.path.exists(base_path+name):
            test = f"{name}.test.csv"
            data = f"{name}.data.csv"
            X, y = get_X_y_from_database(base_path, name, data, test, label)

            dataset_tqdm.set_postfix({"DATABASE": name})
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=seed)
            verbose= True
            seed_tqdm = tqdm(rskf.split(X,y),
                             leave=False,
                             total=n_splits*n_repeats, 
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}') if verbose else rskf.split(X,y)
            i = -1
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

                for j,connections in enumerate(connections_list):
                    graph = AntFeatureGraphMI(seed=seed,connections=connections).compute_graph(X_train, y_train, ("XOR","OR", "AND"))
                    edges_selection[j,i] = len(graph.pheromone_selection)
                    edges_construction[j,i] = len(graph.pheromone_construction)
                nodes[i] = len(graph.nodes)
        else:
            print(f"{name} doesnt' exist")
        

        results.append([name,np.mean(nodes)])
        for index in range(len(connections_list)):
            results[-1].append(np.mean(edges_construction[index,:]))
            results[-1].append(np.mean(edges_selection[index,:]))


        columns = ["Database","Nodes"]
        for connection in connections_list:
            columns.append(f"{connection} - Construction")
            columns.append(f"{connection} - Selection")

        result = pd.DataFrame(results, columns=columns)
        result.to_csv(f"{name}_graph_size.csv",index=False)
    return result



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


df = graph_sizes(datasets,seed,base_path,n_splits,n_repeats,n_intervals)
df.to_csv("graph_size.csv",index=False)