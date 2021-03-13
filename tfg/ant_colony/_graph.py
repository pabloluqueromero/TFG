import networkx as nx
import numpy as np
import random

from tfg.utils import symmetrical_uncertainty
from tfg.encoder import CustomOrdinalFeatureEncoder
from sklearn.preprocessing import LabelEncoder


class AntFeatureGraph:
    def __init__(self,seed):
        self.seed = seed

    def compute_graph(self,X,y,operators):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.label_encoder_ = LabelEncoder()
        X = self.feature_encoder_.fit_transform(X)
        y = self.label_encoder_.fit_transform(y)
        self.nodes = dict(enumerate([
            (j,val)
            for j in range(X.shape[1])
            for val in np.unique(X[:,j])
            ]))
        self.inverse_nodes = {v:k for k,v in self.nodes.items()}
        self.operators = operators
        self.constructed_features = dict()
        self.initial_heuristic = dict({ node:symmetrical_uncertainty(f1=(X[:,value[0]]==value[1]).astype(int),f2=y) for node,value in self.nodes.items()})
        self.pheromone_matrix_selection = dict()
        self.pheromone_matrix_attribute_completion = dict()
        self.allowed_steps = ("CONSTRUCTION","SELECTION")
        return self

    def get_neighbours(self,node,step):
        if step not in self.allowed_steps:
            raise ValueError("Unknown step type: %s, expected one of %s." % (step, self.allowed_steps))
        node_id = self.inverse_nodes[node]
        feature,value = node
        neighbours = []
        if step=="CONSTRUCTION":
            #Optimisable as same variable makes mp sense
            for neighbour,values in filter(lambda x: x[1][0]!=feature,self.nodes.items()):
                for operator in self.operators:
                    edge = frozenset([neighbour,node_id,operator])
                    if edge not in self.pheromone_matrix_attribute_completion:
                        self.pheromone_matrix_attribute_completion[edge] = random.random()
                    neighbours.append((values,operator,self.pheromone_matrix_attribute_completion[edge])) # ((feature_index,value),"OPERATOR",pheromone)
        else:
            for neighbour,values in filter(lambda x: x[0]!=node_id ,self.nodes.items()):
                edge = frozenset([neighbour,node_id])
                if edge not in self.pheromone_matrix_selection:
                    self.pheromone_matrix_selection[edge] = random.random()
                neighbours.append((values,self.pheromone_matrix_selection[edge])) # ((feature_index,value),pheromone)
        return neighbours
                

    def update_pheromone_matrix(self,updated_edges):
        pass

# import pandas as pd
# import os
# def get_X_y(base_path,name,data,test,label):
#     full_data_path = base_path+name+"/"+data
#     full_test_path = base_path+name+"/"+test
#     has_test = os.path.exists(base_path+name+"/"+test)
#     assert pd.read_csv(full_data_path)[label].name == label
#     if has_test:
#         train = pd.read_csv(full_data_path)
#         test = pd.read_csv(full_test_path)
#         df = train.append(test)
    
#     else:
#         df = pd.read_csv(full_data_path)
#     X = df.drop([label],axis=1)
#     y = df[label]
#     return X,y




# base_path, data= "../Dataset/UCIREPO/",["iris","Species"]

# X,y = get_X_y(base_path,data[0],data[0]+".data.csv",test="as",label=data[1])

# afg = AntFeatureGraph(seed=2).compute_graph(X,y,("OR","AND"))
# neighbours = afg.get_neighbours((1,1),"CONSTRUCTION")
# print(neighbours)



