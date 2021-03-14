import pandas as pd
from tfg.ant_colony import Ant
import os
import networkx as nx
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.utils import symmetrical_uncertainty


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

    def get_neighbours(self, node, nodes_to_filter, step):
        if step not in self.allowed_steps:
            raise ValueError("Unknown step type: %s, expected one of %s." % (
                step, self.allowed_steps))
        node_id = self.inverse_nodes[node]
        feature, value = node
        neighbours = []
        pheromones = []
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
                        (neighbour,values, operator))# (id,(feature_index,value),"OPERATOR")
                    pheromones.append(self.pheromone_matrix_attribute_completion[edge])
        else:
            for neighbour, values in filter(lambda x: x[0] not in nodes_to_filter and x[0] != node_id, self.nodes.items()):
                edge = frozenset([neighbour, node_id])
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                # (id,(feature_index,value),pheromone)
                neighbours.append((neighbour,values))
                pheromones.append(self.pheromone_matrix_selection[edge])
        return neighbours,np.array(pheromones)

    def get_initial_nodes(self):
        initial = []
        heuristic = []
        pheromone = []
        for node, values in self.nodes.items():
            heuristic.append(self.initial_heuristic[node])
            pheromone.append(self.initial_pheromone[node])
            initial.append((node,values))
        return initial,np.array(pheromone),np.array(heuristic)

    def update_pheromone_matrix_evaporation(self, evaporation_rate):
        update_factor = (1-evaporation_rate)
        self.pheromone_matrix_selection = {k:v*update_factor for k,v in self.pheromone_matrix_selection.items()}
        self.pheromone_matrix_attribute_completion = {k:v*update_factor for k,v in self.pheromone_matrix_attribute_completion.items()}
        self.initial_heuristic =  {k:v*update_factor for k,v in self.initial_heuristic.items()}

    def intensify(self,features,intensification_factor):
        previous = None
        for feature in features:
            if isinstance(feature,DummyFeatureConstructor):
                next_node = self.inverse_nodes[(feature.feature_index,None)]
                if previous is None:
                    self.initial_pheromone[next_node] += intensification_factor
                else:
                    self.pheromone_matrix_selection[frozenset([previous,next_node])] += intensification_factor
            else:
                operands = feature.operands
                next_node = self.inverse_nodes[(operands[0].feature_index,operands[0].value)]
                if previous is None:
                    self.initial_pheromone[next_node] += intensification_factor
                else:
                    self.pheromone_matrix_selection[frozenset([previous,next_node])] += intensification_factor
            
                previous = next_node
                next_node = self.inverse_nodes[(operands[1].feature_index,operands[1].value)]
                edge = frozenset([previous,next_node,feature.operator])
                self.pheromone_matrix_attribute_completion[edge] += intensification_factor
            previous = next_node
        return
