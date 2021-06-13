import pandas as pd
import os
import numpy as np
import random
import networkx as nx

from tfg.optimization.ant_colony import Ant
from tfg.feature_construction import DummyFeatureConstructor
from tfg.utils import symmetrical_uncertainty
from tfg.utils import symmetrical_uncertainty_two_variables, symmetrical_uncertainty_class_conditioned
from tfg.feature_construction._constructor import create_feature
import math as m

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
        self.original_ids = [self.inverse_nodes[(i,None)] for i in range(X.shape[1])]
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
            for neighbour_id, values in filter(lambda x: x[1][1] is not None  and x[0]!=node_id, self.nodes.items()):
                for operator in (self.operators if values[0]!=feature else ['OR']):
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
            #Dont add loops if it is an original vairable
            for neighbour_id, values in filter(lambda x: x[0] not in nodes_to_filter and not(x[0] == node_id and value is None), self.nodes.items()):
                edge = frozenset([neighbour_id, node_id])
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                neighbours.append((neighbour_id,values))
                pheromones.append(self.pheromone_matrix_selection[edge])
        return neighbours,np.array(pheromones)

    def get_initial_nodes(self,selected_nodes):
        '''Initial nodes
           Note: it is necessary to sort the ids so that they match the order of the arrays in the pheromon and heuristic arrays.
                 Dicts do not guarantee to follow insertion order (optionally this could be replaced with an OrderedDict'''
        nodes_pheromone_heuristic = zip(sorted(list(self.nodes.items()),key=lambda x: x[0]), self.initial_pheromone, self.initial_heuristic) 
        return list(zip(*(filter(lambda x: x[0][0] not in selected_nodes, nodes_pheromone_heuristic))))

    def update_pheromone_matrix_evaporation(self, evaporation_rate):
        '''Evaporate the pheromones by the given factor'''
        update_factor = (1-evaporation_rate)
        self.pheromone_matrix_selection = {k:v*update_factor for k,v in self.pheromone_matrix_selection.items()}
        self.pheromone_matrix_attribute_completion = {k:v*update_factor for k,v in self.pheromone_matrix_attribute_completion.items()}
        self.initial_pheromone =  self.initial_pheromone*update_factor

    def intensify(self,features,intensification_factor,ant_score=1,use_initials=False):
        '''Intensify the path of followed by the given ant'''
        previous = None
        intensification_factor*=ant_score
        for feature in features:
            if isinstance(feature,DummyFeatureConstructor):
                if use_initials:
                    continue
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
        
    def get_original_ids(self):
        return self.original_ids




class AntFeatureGraphMI:
    def __init__(self, seed, connections=2):
        self.seed = seed
        self.connections = connections    
        self.allowed_steps = ("CONSTRUCTION", "SELECTION")

    def _initialize_graph(self,X,y,k=2):
        self.selection_graph = nx.MultiDiGraph()
        self.selection_graph.add_nodes_from(self.initial_graph.nodes)
        self.pheromone_selection = dict()

        self.construction_graph = nx.MultiDiGraph()
        self.construction_graph.add_nodes_from(self.initial_graph.nodes)
        self.pheromone_construction = dict()

        self.neighbour_features_ = dict()
        '''PRUNNING'''
        for i in range(X.shape[1]):
            mi = []
            for j in range(X.shape[1]):
                if i==j:
                    continue
                mi.append((j,symmetrical_uncertainty_two_variables(X[:,j],X[:,i],y)))
            mi = sorted(mi,key = lambda x:x[1],reverse=True) # The greater the mutual information score the more correlation which we want to avoid
            self.neighbour_features_[i] = list(zip(*mi[:k]))[0]
        
        '''MULTIGRAPH CONSTRUCTION'''
        for feature in range(X.shape[1]):
            '''Original'''
            node_id = self.inverse_nodes[(feature,None)]

            #Allow jumps to any other original feature
            for neighbour_feature in range(X.shape[1]):
                neighbour_id = self.inverse_nodes[(neighbour_feature,None)]
                edge = frozenset([neighbour_id,node_id])
                self.selection_graph.add_edge(node_id,neighbour_id)
                self.pheromone_selection[edge] = random.random()

            #Add neighbours from other features
            for neighbour_feature in self.neighbour_features_[feature]:
                for neighbour_feature_value in self.unique_values_ferature[neighbour_feature]:
                    neighbour_id = self.inverse_nodes[(neighbour_feature,neighbour_feature_value)]
                    edge = frozenset([neighbour_id, node_id])
                    self.selection_graph.add_edge(node_id,neighbour_id)
                    self.pheromone_selection[edge] = random.random()

            '''Logical'''
            for value in self.unique_values_ferature[feature]:
                node_id = self.inverse_nodes[(feature,value)]
                '''SELECTION'''
                #Add all operand nodes (same feature) - including nodex_id
                for value_neighbour in self.unique_values_ferature[feature]:
                    neighbour_id = self.inverse_nodes[(feature,value_neighbour)]
                    values = self.nodes[neighbour_id]
                    edge = frozenset([neighbour_id, node_id])
                    self.selection_graph.add_edge(node_id,neighbour_id)
                    self.pheromone_selection[edge] = random.random()

                #Add neighbours from other features
                for neighbour_feature in self.neighbour_features_[feature]:
                    for neighbour_feature_value in self.unique_values_ferature[neighbour_feature]:
                        neighbour_id = self.inverse_nodes[(neighbour_feature,neighbour_feature_value)]
                        edge = frozenset([neighbour_id, node_id])
                        self.selection_graph.add_edge(node_id,neighbour_id)
                        self.pheromone_selection[edge] = random.random()
                    
                # --> We don't allow jumpt to other features anymore
                #Allow jumps to any other original feature
                for neighbour_feature in range(X.shape[1]):
                    neighbour_id = self.inverse_nodes[(neighbour_feature,None)]
                    edge = frozenset([neighbour_id,node_id])
                    self.selection_graph.add_edge(node_id,neighbour_id)
                    self.pheromone_selection[edge] = random.random()


                '''CONSTRUCTION'''

                #Other features
                for neighbour_feature in self.neighbour_features_[feature]:
                    for operator in self.operators:
                        for value_neighbour in self.unique_values_ferature[neighbour_feature]:
                            neighbour_id = self.inverse_nodes[(neighbour_feature,value_neighbour)]
                            edge = frozenset([neighbour_id, node_id, operator])
                            self.construction_graph.add_edge(node_id,neighbour_id,operator = operator)
                            self.pheromone_construction[edge] = random.random()

                #Within same feature except (node_id,node_id,OR)
                for value_neighbour in  self.unique_values_ferature[feature]:
                    neighbour_id = self.inverse_nodes[(feature,value_neighbour)]
                    if neighbour_id == node_id:
                        continue
                    operator="OR"
                    edge = frozenset([neighbour_id, node_id, operator])
                    self.construction_graph.add_edge(node_id,neighbour_id, operator = operator)
                    self.pheromone_construction[edge] = random.random()

    def _initialize_nodes(self,X,y):
        self.initial_graph = nx.Graph()
        self.nodes = dict()
        self.pheromone_initial = dict()

        self.unique_values_ferature = {j:np.unique(X[:,j]) for j in range(X.shape[1])}
        for j in range(X.shape[1]):
            node_id = len(self.nodes)
            self.nodes[node_id] = (j, None)
            self.pheromone_initial[node_id] = random.random()
            su = symmetrical_uncertainty(f1=X[:,j], f2=y)
            self.initial_graph.add_node(node_id,value = (j,None), heuristic =su)
            for node_id,value in enumerate(self.unique_values_ferature[j], start = len(self.nodes)):
                self.nodes[node_id] = (j, value)
                su = symmetrical_uncertainty(f1=(X[:, j] == value), f2=y)
                self.initial_graph.add_node(node_id,value = (j,value),heuristic =su)
                self.pheromone_initial[node_id] = random.random()
        self.inverse_nodes = {v: k for k, v in self.nodes.items()}
        self.original_ids = [self.inverse_nodes[(i,None)] for i in range(X.shape[1])]

            
    def compute_graph(self, X, y, operators):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.operators = operators
        self._initialize_nodes(X,y)
        self._initialize_graph(X,y,k = self.connections)
        return self

    def get_neighbours(self, node, nodes_to_filter, step, percentage = 1):
        if step not in self.allowed_steps:
            raise ValueError("Unknown step type: %s, expected one of %s." % (
                step, self.allowed_steps))
        node_id = self.inverse_nodes[node]
        feature, value = node
        neighbours = []
        pheromones = []
        if step == "CONSTRUCTION":
            for neighbour_id in self.construction_graph.neighbors(node_id):
                for operator in map(lambda x: x["operator"],self.construction_graph.get_edge_data(node_id,neighbour_id).values()):
                    edge = frozenset([neighbour_id, node_id, operator])
                    if edge in nodes_to_filter:
                        continue
                    neighbours.append((neighbour_id,self.nodes[neighbour_id],operator))
                    pheromones.append(self.pheromone_construction[edge])

        elif step == "SELECTION":
            for neighbour_id in filter(lambda x: x not in nodes_to_filter,self.selection_graph.neighbors(node_id)):
                edge = frozenset([neighbour_id, node_id])
                neighbours.append((neighbour_id,self.nodes[neighbour_id]))
                pheromones.append(self.pheromone_selection[edge])
        else:
            raise ValueError("Unknown step")
        if percentage < 1:
            pheromones = np.array(pheromones)
            indexes = np.random.choice(np.arange(len(neighbours)),m.ceil(len(neighbours)*percentage), p = pheromones /pheromones.sum())
            return [neighbours[index] for index in indexes], pheromones[indexes]
        return neighbours, np.array(pheromones)

    def get_initial_nodes(self,selected_nodes,percentage=1):
        data = self.initial_graph.nodes(data=True)
        nodes= []
        pheromones = []
        heuristic = []
        for i,node_data in enumerate(data):
            node,attributes = node_data
            if node not in selected_nodes:
                nodes.append((node,attributes["value"]))
                heuristic.append(attributes["heuristic"])
                pheromones.append(self.pheromone_initial[node])
                
        if percentage < 1:
            pheromones = np.array(pheromones)
            indexes = np.random.choice(np.arange(len(nodes)),m.ceil(len(nodes)*percentage), p = pheromones /pheromones.sum())
            return [nodes[index] for index in indexes], pheromones[indexes], np.array(heuristic)[indexes]
        return nodes,np.array(pheromones),np.array(heuristic)

    def reset_pheromones(self):
        self.pheromone_construction = { k: random.random() for k in self.pheromone_construction}
        self.pheromone_selection = { k: random.random() for k in self.pheromone_selection}
        self.pheromone_initial = { k: random.random() for k in self.pheromone_initial}

    def update_pheromone_matrix_evaporation(self, evaporation_rate):
        update_factor = (1-evaporation_rate)
        self.pheromone_selection = {k:v*update_factor for k,v in self.pheromone_selection.items()}
        self.pheromone_construction = {k:v*update_factor for k,v in self.pheromone_construction.items()}
        self.pheromone_initial = {k:v*update_factor for k,v in self.pheromone_initial.items()}


    def intensify(self,features,intensification_factor,ant_score=1,use_initials=False):
        '''Intensify the path followed by the given ant'''
        previous = None
        intensification_factor*=ant_score
        if use_initials:
            index=0
            feature= features[index]
            while isinstance(feature,DummyFeatureConstructor):
                index+=1
                if index >= len(features):
                    break
                feature= features[index]
            features = features[index:]

        for feature in features:
            if isinstance(feature,DummyFeatureConstructor):
                if use_initials:
                    continue
                next_node = self.inverse_nodes[(feature.feature_index,None)]
                if previous is None:
                    self.pheromone_initial[next_node] += intensification_factor
                else:
                    if frozenset([previous,next_node]) in self.pheromone_selection:
                        #When incremental approach split point
                        self.pheromone_selection[frozenset([previous,next_node])] += intensification_factor
            else:
                operands = feature.operands
                next_node = self.inverse_nodes[(operands[0].feature_index,operands[0].value)]
                if previous is None:
                    self.pheromone_initial[next_node] += intensification_factor
                else:
                    if frozenset([previous,next_node]) in self.pheromone_selection:
                        #When incremental approach split point
                        self.pheromone_selection[frozenset([previous,next_node])] += intensification_factor
            
                previous = next_node
                next_node = self.inverse_nodes[(operands[1].feature_index,operands[1].value)]
                edge = frozenset([previous,next_node,feature.operator])
                self.pheromone_construction[edge] += intensification_factor
            previous = next_node
        return
    
    def get_random_node(self):
        index = random.randint(0,len(self.nodes)-1)
        return index,self.nodes[index], self.initial_graph.nodes[index]["heuristic"]

    def max_initial(self):
        best_feature = max(self.pheromone_initial, key = lambda x: self.pheromone_initial[x])
        return best_feature,self.nodes[best_feature], self.initial_graph.nodes[best_feature]["heuristic"]

    def get_original_ids(self):
        return self.original_ids

    def get_rank(self,):
        all_feature_constructors = []
        constructed_features = set()
        for node_id in self.construction_graph.nodes:
            node_element = self.nodes[node_id]
            for neighbour_node_id in self.construction_graph.neighbors(node_id):
                node_neighbour = self.nodes[neighbour_node_id]
                for operator in self.operators:
                    feature = create_feature(operator = operator, operands = [node_element, node_neighbour])
                    hashed_feature = hash(feature)
                    if hashed_feature in constructed_features:
                        continue
                    all_feature_constructors.append(feature)
                    constructed_features.add(hashed_feature)
        return all_feature_constructors

