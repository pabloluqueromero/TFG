import pandas as pd
from tfg.ant_colony import Ant
import os
import networkx as nx
import numpy as np
import random

from tfg.utils import symmetrical_uncertainty
from tfg.encoder import CustomOrdinalFeatureEncoder
from sklearn.preprocessing import LabelEncoder


class AntFeatureGraph:
    def __init__(self, seed):
        self.seed = seed

    def compute_graph(self, X, y, operators):
        random.seed(self.seed)
        self.nodes = dict(enumerate([
            (j, val)
            for j in range(X.shape[1])
            for val in np.unique(X[:, j])
        ]))
        # Add original variables
        self.nodes.update(
            dict(enumerate([(j, None) for j in range(X.shape[1])], start=len(self.nodes))))
        self.inverse_nodes = {v: k for k, v in self.nodes.items()}
        self.operators = operators
        self.initial_heuristic = dict({
            node:
            symmetrical_uncertainty(
                f1=(X[:, value[0]] == value[1]).astype(int), f2=y)
            if value[1] is not None else
            symmetrical_uncertainty(f1=X[:, value[0]], f2=y)
            for node, value in self.nodes.items()
        })
        self.initial_pheromone = dict(
            {node: random.random() for node in self.nodes.keys()})
        self.pheromone_matrix_selection = dict()
        self.pheromone_matrix_attribute_completion = dict()
        self.allowed_steps = ("CONSTRUCTION", "SELECTION")
        return self

    def get_neighbours(self, node,nodes_to_filter, step):
        if step not in self.allowed_steps:
            raise ValueError("Unknown step type: %s, expected one of %s." % (
                step, self.allowed_steps))
        node_id = self.inverse_nodes[node]
        feature, value = node
        neighbours = []
        if step == "CONSTRUCTION":
            # Optimisable could save index
            for neighbour, values in filter(lambda x: x[1][0] != feature and x[1][1] != None, self.nodes.items()):
                for operator in self.operators:
                    edge = frozenset([neighbour, node_id, operator])
                    if edge in nodes_to_filter:
                        continue
                    if edge not in self.pheromone_matrix_attribute_completion:
                        self.pheromone_matrix_attribute_completion[edge] = random.random()
                    neighbours.append(
                        (neighbour,values, operator, self.pheromone_matrix_attribute_completion[edge]))# (id,(feature_index,value),"OPERATOR",pheromone)
        else:
            for neighbour, values in filter(lambda x: x[0] not in nodes_to_filter and x[0] != node_id, self.nodes.items()):
                edge = frozenset([neighbour, node_id])
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                # (id,(feature_index,value),pheromone)
                neighbours.append((neighbour,values, self.pheromone_matrix_selection[edge]))
        return neighbours

    def get_initial_nodes(self):
        initial = []
        for node, values in self.nodes.items():
            heuristic = self.initial_heuristic[node]
            pheromone = self.initial_pheromone[node]
            initial.append((node,values, heuristic, pheromone))
        return initial

    def update_pheromone_matrix(self, updated_edges):
        pass














######################################################################
#
#
#                               TESTS    
#               
#
######################################################################
def get_X_y(base_path, name, data, test, label):
    full_data_path = base_path+name+"/"+data
    full_test_path = base_path+name+"/"+test
    has_test = os.path.exists(base_path+name+"/"+test)
    assert pd.read_csv(full_data_path)[label].name == label
    if has_test:
        train = pd.read_csv(full_data_path)
        test = pd.read_csv(full_test_path)
        df = train.append(test)

    else:
        df = pd.read_csv(full_data_path)
    X = df.drop([label], axis=1)
    y = df[label]
    return X, y


base_path, data = "../Dataset/UCIREPO/", ["iris", "Species"]

X, y = get_X_y(base_path, data[0], data[0] +
               ".data.csv", test="as", label=data[1])

feature_encoder_ = CustomOrdinalFeatureEncoder()
label_encoder_ = LabelEncoder()
X = feature_encoder_.fit_transform(X)
y = label_encoder_.fit_transform(y)
afg = AntFeatureGraph(seed=2).compute_graph(X, y, ("OR", "AND"))
# neighbours = afg.get_neighbours((1, 1), "CONSTRUCTION")
# print(neighbours)
random.seed(2222)
Ant(ant_id=1, alpha=0.5, beta=0.3).run(X, y, afg,random)
