import pandas as pd
import os
import networkx as nx
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder

from tfg.ant_colony import Ant
from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.utils import symmetrical_uncertainty
from tfg.utils import mutual_information_class_conditioned

###################################################################################
#
#
#                   GRAPH implementations for the ACFCS algorithm
#                   
#                  1. AntFeatureGraph: full graph
#                  2. AntFeatureGraphMI: reduced graph using 
#                                        Mutual Information as heuristic 
#
###################################################################################
class AntFeatureGraph:    
    """ Graph for the ACFCS algorithm.
    
    Fully connected graph containing the edges for the ants to explore and the pheromone matrix.
    The graph:
        - Two types of nodes (each node has a unique ID):
            1. Nodes representing an original future represented by (feature_index,None)
            2. Nodes representing a value of a feature which will be part of a logical feature (feature_index,value)
        - Three types of edges:
            1. Initial edges, connecting every node to a "logical" node that represents the initial state with no features selected
               The heuristic for each of this edeges is computed once at the beginning
            2. Construction edges. Between nodes representing a value of a feature. There are n of them (one for each operator) (Only the pheromone is stored)
            3. Selection nodes. Between all the nodes. (Only the pheromone is stored)
    Although the edges are lazily stored and computed there is no efficient way to quickly directly get the neighbours as storage may exceed memory 
    allowance therefore neighbours are computed every time.

    Parameters
    ----------
    seed : float or None
        Seed used for the generation of random values for the pheromone trail.
                  
    Attributes
    ----------
    nodes : dict
        Dict mapping node id to relevant info (feature_index, value)
    inverse_nodes: dict
        Reverse dict nodes

    initial_heuristic : array-like of shape (n_nodes,)
        Contains the SU for each edge (initial node, node_i) individually evaluated

    initial_pheromone : array-like of shape (n_nodes,)
        Contains the SU for each edge (initial node, node_i)

    pheromone_matrix_selection : dict
        Dictionary with the pheromone value for each edge.
        Keys are frozensets because of the symmetrical matrix.
        { frozentset(node_i,node_j ): float}

    pheromone_matrix_attribute_completion : dict
        Dictionary with the pheromone value for each edge between nodes representing (feature,value).
        Keys are frozensets because of the symmetrical matrix.
        { frozentset(node_i,node_j,operator): float}

    allowed_steps : tuple ("CONSTRUCTION","SELECTION")
        Strategy for the neighbour generation

    operators: tuple of str
        Operator used, noramlly ('AND', 'OR', 'XOR')
    
    """
    def __init__(self, seed):
        self.seed = seed

    def _initialize_initial_matrix(self,X,y):
        '''Initializes the initial_heuristic and the initial_pheromone'''
        self.initial_heuristic = np.empty(len(self.nodes))
        self.initial_pheromone = np.random.rand(len(self.nodes))
        for node, value in self.nodes.items():
            if value[1] is None:
                su = symmetrical_uncertainty(f1=X[:, value[0]], f2=y)
            else:
                su = symmetrical_uncertainty(f1=(X[:, value[0]] == value[1]).astype(int), f2=y)
            self.initial_heuristic[node] = su
       
    def _initialize_pheromone_matrix(self):
        '''Initializes the dictionaries for the pheromone trails'''
        self.pheromone_matrix_selection = dict()
        self.pheromone_matrix_attribute_completion = dict()

    def _initialize_nodes(self,X):
        '''Initializes the nodes, it adds the original features at the end'''
        self.nodes = dict()
        for j in range(X.shape[1]):
            unique_values = np.unique(X[:, j])
            for node_id,val in enumerate(unique_values,start = len(self.nodes)):
                self.nodes[node_id] = (j, val)
        # Add original variables
        self.nodes.update(dict(enumerate([(j, None) for j in range(X.shape[1])], start=len(self.nodes))))
        
    def compute_graph(self, X, y, operators):
        '''Initializes the nodes, it adds the original features at the end'''
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._initialize_nodes(X)
        self.inverse_nodes = {v: k for k, v in self.nodes.items()}
        self.operators = operators
        self._initialize_initial_matrix(X,y)
        self._initialize_pheromone_matrix()
        self.allowed_steps = ("CONSTRUCTION", "SELECTION")
        return self

    def get_neighbours(self, node, nodes_to_filter, step):
        '''Get neighbors accoring to the selected strategy CONSTRUCTION or SELECTION, returns the neighbours and the pheromone valu for each of the edgesbetween them'''
        if step not in self.allowed_steps:
            raise ValueError("Unknown step type: %s, expected one of %s." % (
                step, self.allowed_steps))
        node_id = self.inverse_nodes[node]
        feature, value = node
        neighbours = []
        pheromones = []
        if step == "CONSTRUCTION":
            # Cannot construct with the same feature or with an original variable.
            for neighbour_id, values in filter(lambda x: x[1][0] != feature and x[1][1] != None, self.nodes.items()):
                for operator in self.operators:
                    edge = frozenset([neighbour_id, node_id, operator])
                    if edge in nodes_to_filter:
                        continue
                    if edge not in self.pheromone_matrix_attribute_completion:
                        self.pheromone_matrix_attribute_completion[edge] = random.random()
                    neighbours.append(
                        (neighbour_id,values, operator))# (id,(feature_index,value),"OPERATOR")
                    pheromones.append(self.pheromone_matrix_attribute_completion[edge])
        else:
            #Cannot select the same node or nodes that have already been selected (normally these can only be original features as constructable nodes 
            #can appear more than once)
            for neighbour_id, values in filter(lambda x: x[0] not in nodes_to_filter and x[0] != node_id, self.nodes.items()):
                edge = frozenset([neighbour_id, node_id])
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                neighbours.append((neighbour_id,values))
                pheromones.append(self.pheromone_matrix_selection[edge])
        return neighbours,np.array(pheromones)

    def get_initial_nodes(self):
        '''Initial nodes
           Note: it is necessary to sort the ids so that they match the order of the arrays in the pheromon and heuristic arrays.
                 Dicts do not guarantee to follow insertion order (optionally this could be replaced with an OrderedDict'''
        return sorted(list(self.nodes.items()),key=lambda x:x[0]),self.initial_pheromone,self.initial_heuristic

    def update_pheromone_matrix_evaporation(self, evaporation_rate):
        '''Evaporate the pheromones by the given factor'''
        update_factor = (1-evaporation_rate)
        self.pheromone_matrix_selection = {k:v*update_factor for k,v in self.pheromone_matrix_selection.items()}
        self.pheromone_matrix_attribute_completion = {k:v*update_factor for k,v in self.pheromone_matrix_attribute_completion.items()}
        self.initial_pheromone =  self.initial_pheromone*update_factor

    def intensify(self,features,intensification_factor):
        '''Intensify the path of followed by the given ant'''
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

class AntFeatureGraphMI:
    def __init__(self, seed):
        self.seed = seed

    def _initialize_initial_matrix(self,X,y):
        self.initial_heuristic = np.empty(len(self.nodes))
        self.initial_pheromone = np.random.rand(len(self.nodes))
        for node, value in self.nodes.items():
            if value[1] is None:
                su = symmetrical_uncertainty(f1=X[:, value[0]], f2=y)
            else:
                su = symmetrical_uncertainty(f1=(X[:, value[0]] == value[1]), f2=y)
            self.initial_heuristic[node] = su
            self.initial_pheromone[node] = random.random()
       
    def _initialize_pheromone_matrix(self):
        # normalized_mutual_info_score
        self.pheromone_matrix_selection = dict()
        self.pheromone_matrix_attribute_completion = dict()

    def _initialize_nodes(self,X):
        self.nodes = dict()
        self.nodes_per_feature = dict()
        for j in range(X.shape[1]):
            unique_values = np.unique(X[:, j])
            self.nodes_per_feature[j] = (len(self.nodes),unique_values.shape[0]) #(first_id, total)
            for node_id,val in enumerate(unique_values,start = len(self.nodes)):
                self.nodes[node_id] = (j, val)
        # Add original variables
        self.nodes.update(dict(enumerate([(j, None) for j in range(X.shape[1])], start=len(self.nodes))))
        
    def _initialize_neighbours_info(self,X,y,k=2):
        '''Add neighbour features based on mutual information'''
        self.neighbour_features_ = dict()
        for i in range(X.shape[1]):
            mi = []
            for j in range(X.shape[1]):
                if i==j:
                    continue
                mi.append((j,mutual_information_class_conditioned(X[:,j],X[:,i],y)))
            mi = sorted(mi,key = lambda x:x[1],reverse=False) # The greater the mutual information score the more correlation which we want to avoid
            self.neighbour_features_[i]=list(zip(*mi[:k]))[0]
        


    def compute_graph(self, X, y, operators):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._initialize_nodes(X)
        self.inverse_nodes = {v: k for k, v in self.nodes.items()}
        self.operators = operators
        self._initialize_neighbours_info(X,y)
        self._initialize_initial_matrix(X,y)
        self._initialize_pheromone_matrix()
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
            for neighbour_feature in self.neighbour_features_[feature]:
                for operator in self.operators:
                    neighbour_feature_first_index = self.nodes_per_feature[neighbour_feature][0]
                    neighbour_feature_n_nodes = self.nodes_per_feature[neighbour_feature][1]
                    for neighbour_id in range(neighbour_feature_first_index,neighbour_feature_first_index+neighbour_feature_n_nodes):
                        values = self.nodes[neighbour_id]
                        edge = frozenset([neighbour_id, node_id, operator])
                        if edge in nodes_to_filter:
                            continue
                        if edge not in self.pheromone_matrix_attribute_completion:
                            self.pheromone_matrix_attribute_completion[edge] = random.random()
                        neighbours.append( (neighbour_id,values, operator))# (id,(feature_index,value),"OPERATOR")
                        pheromones.append(self.pheromone_matrix_attribute_completion[edge])
        else:
            #Adding neighbours that are from the same feature
            neighbour_feature_first_index = self.nodes_per_feature[feature][0]
            neighbour_feature_n_nodes = self.nodes_per_feature[feature][1]
            #Add all individual
            for neighbour_id in range(neighbour_feature_first_index,neighbour_feature_n_nodes+neighbour_feature_first_index):
                values = self.nodes[neighbour_id]
                edge = frozenset([neighbour_id, node_id])
                if neighbour_id in nodes_to_filter:
                    continue
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                neighbours.append( (neighbour_id,values))# (id,(feature_index,value),"OPERATOR")
                pheromones.append(self.pheromone_matrix_selection[edge])

            #Add neighbours from other features
            for neighbour_feature in self.neighbour_features_[feature]:                
                neighbour_feature_first_index = self.nodes_per_feature[neighbour_feature][0]
                neighbour_feature_n_nodes = self.nodes_per_feature[neighbour_feature][1]
                for neighbour_id in range(neighbour_feature_first_index,neighbour_feature_n_nodes+neighbour_feature_first_index):
                    values = self.nodes[neighbour_id]
                    edge = frozenset([neighbour_id, node_id])
                    if neighbour_id in nodes_to_filter:
                        continue
                    if edge not in self.pheromone_matrix_selection:
                        self.pheromone_matrix_selection[edge] = random.random()
                    neighbours.append( (neighbour_id,values))# (id,(feature_index,value),"OPERATOR")
                    pheromones.append(self.pheromone_matrix_selection[edge])
                
            #Allow jumps to any other original feature
            for neighbour_feature in range(len(self.nodes_per_feature)):
                neighbour_id = self.inverse_nodes[(neighbour_feature,None)]
                if neighbour_id in nodes_to_filter or neighbour_id ==node_id:
                    continue
                edge = frozenset([neighbour_id,node_id])
                values = self.nodes[neighbour_id]
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                neighbours.append((neighbour_id,values))
                pheromones.append(self.pheromone_matrix_selection[edge])            
        return neighbours,np.array(pheromones)

    def get_initial_nodes(self):
        return sorted(list(self.nodes.items()),key=lambda x: x[0]), self.initial_pheromone, self.initial_heuristic

    def update_pheromone_matrix_evaporation(self, evaporation_rate):
        update_factor = (1-evaporation_rate)
        self.pheromone_matrix_selection = {k:v*update_factor for k,v in self.pheromone_matrix_selection.items()}
        self.pheromone_matrix_attribute_completion = {k:v*update_factor for k,v in self.pheromone_matrix_attribute_completion.items()}
        self.initial_heuristic *= update_factor

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
