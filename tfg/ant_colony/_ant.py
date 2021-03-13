import threading
import numpy as np

from tfg.naive_bayes import NaiveBayes
from tfg.feature_construction import DummyFeatureConstructor
from tfg.feature_construction._feature import FeatureOperand
from tfg.feature_construction._constructor import create_feature
from tfg.utils import compute_sufs

class Ant:
    def __init__(self, ant_id, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.ant_id = ant_id

    def choose_next(self, probabilities, random_generator):
        n = random_generator.uniform(0.0, 1.0)
        cumulative_sum = 0
        index = 0
        while cumulative_sum < n:
            cumulative_sum += probabilities[index]
            index += 1
        return index-1

    def explore(self, X, y, graph, random_generator):
        self.current_features = []
        selected_nodes = set()
        constructed_nodes = set()
        classifier = NaiveBayes(encode_data=False)
        current_score = np.NINF
        score = 0

        initial = graph.get_initial_nodes()
        probabilities = np.array([np.power(pheromone,self.alpha)*np.power(heuristic,self.beta)
                                  for _, values, heuristic, pheromone in initial])
        probabilities /= probabilities.sum()
        index = self.choose_next(probabilities, random_generator)

        current_su = 0
        node_id, selected_node,su = initial[index][:3]
        is_fitted = False
        feature_constructor = None
        while True:
            current_score = score
            if selected_node[1] is None:
                # Original Feature
                feature_constructor = DummyFeatureConstructor(selected_node[0])
                
                if is_fitted:
                    classifier.add_features(
                        feature_constructor.transform(X), y)
                else:
                    classifier.fit(feature_constructor.transform(X), y)
                    is_fitted = True
                features = np.concatenate([f.transform(X) for f in self.current_features]+[feature_constructor.transform(X)],axis=1)
                score = classifier.leave_one_out_cross_val(features, y,fit=False)
                if score <= current_score:
                    break
                current_su = su
                selected_nodes.add(node_id)
                self.current_features.append(feature_constructor)
                current_score = score
            else:
                # Need to construct next feature and compute heuristic value for the feature
                neighbours = graph.get_neighbours(
                    selected_node, constructed_nodes, step="CONSTRUCTION")
                # Compute heuristic
                su = []
                for neighbour in neighbours:
                    operator = neighbour[2] # Get operator
                    operands = [selected_node,neighbour[1]] #Get operands (index,value)
                    feature = create_feature(operator,operands)
                    su.append(compute_sufs(current_su,[f.transform(X).flatten() for f in self.current_features],feature.transform(X).flatten(),y,minimum=0))
                
                probabilities = np.array([(np.power(neighbours[i][3],self.alpha))*np.power(su[i],self.beta)
                                        for i in range(len(neighbours))])
                probabilities /= probabilities.sum()
                index = self.choose_next(probabilities, random_generator)
                
                su = su[index]
                feature_constructor = create_feature(neighbours[index][2],[selected_node,neighbours[index][1]])
                
                if is_fitted:
                    classifier.add_features(
                        feature_constructor.transform(X), y)
                else:
                    classifier.fit(feature_constructor.transform(X), y)
                    is_fitted = True
                features = np.concatenate([f.transform(X) for f in self.current_features]+[feature_constructor.transform(X)],axis=1)   
                score = classifier.leave_one_out_cross_val(features, y)
                if score <= current_score:
                    break
                current_su = su
                constructed_nodes.add(frozenset((node_id,neighbours[index][0],neighbours[index][2])))
                self.current_features.append(feature_constructor)
                current_score = score

                node_id,selected_node = neighbours[index][:2]

            #Select next
            neighbours = graph.get_neighbours(
                selected_node, selected_nodes, step="SELECTION")
            # Compute heuristic
            su = []
            for neighbour in neighbours:
                if neighbour[1][1] is None:
                    #Original variable
                    su.append(compute_sufs(current_su,[f.transform(X).flatten() for f in self.current_features],X[:, neighbour[1][0]],y,minimum=0))
                else:
                    #This is a temporal variable that will not be finally selected but only used to calculate the heuristic
                    su.append(compute_sufs(current_su,[f.transform(X).flatten() for f in self.current_features],X[:, neighbour[1][0]] == neighbour[1][1],y,minimum=0))
            
            probabilities = np.array([(np.power(neighbours[i][2],self.alpha))*np.power(su[i],self.beta)
                                    for i in range(len(neighbours))])
            try:
                probabilities /= probabilities.sum()
            except:
                print("aqui")
                pass
            index = self.choose_next(probabilities, random_generator)
            
            su = su[index]
            node_id,selected_node = neighbours[index][:2]
        self.final_score = current_score
        return self.final_score

    def run(self, X, y, graph, random_generator):
        # print(f"Ant [{self.ant_id}] running in thread [{threading.get_ident()}]")
        return self.explore(X, y, graph,random_generator)
