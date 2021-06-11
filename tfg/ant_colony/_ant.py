import concurrent
import numpy as np
import random
import math

from tfg.naive_bayes import NaiveBayes
from tfg.feature_construction import DummyFeatureConstructor, FeatureOperand
from tfg.feature_construction._constructor import create_feature
from tfg.utils import append_column_to_numpy, backward_search, compute_sufs, hash_features


class Ant:
    """ Basic building block for the Ant Colony Feature Construction and Selection algorithm.

    Logical representation of an ant which performs a heuristic exploration on the graph based on previous knowledge to
    optimise the subset feature selection and creation. 

    Parameters
    ----------
    alpha : float
        Importance accorded to  the pheromone trail

    beta : float
        Importance accorded to  the dynamically computed heuristic

    ant_id : int
        Parameter to uniquely identify the ant.

    Attributes
    ----------
    current_features : array-like of Features
        Features obtained by the end of the exploration

    final_score : float
        Leave one out croos-validation accuracy score obtained with the selected feature subset    
    """

    def __init__(self, ant_id, alpha, beta, metric, use_initials=False, cache_loo = dict(), cache_heuristic = dict(), step = 1):
        self.alpha = alpha
        self.beta = beta
        self.ant_id = ant_id
        self.metric = metric
        self.use_initials = use_initials
        self.cache_loo = cache_loo
        self.cache_heuristic = cache_heuristic
        self.step = step

    def choose_next(self, probabilities, random_generator):
        '''Selects index based on roulette wheel selection'''
        n = random_generator.uniform(0.0, 1.0)
        cumulative_sum = 0
        index = 0
        while cumulative_sum < n:
            cumulative_sum += probabilities[index]
            index += 1
        return index-1
        # return np.random.choice(np.arange(len(probabilities)),1,p = probabilities)[0]

    def compute_probability(self, pheromones, heuristics):
        '''Computes the probability based on the formula
           p(edge_ij) = normalized(pheromone_ij**alpha * heuristic_ij**beta)

           Note:
           Some constructed features may produce features with no positive values if 
           the constructed features contains a combination of values that is not present in the database
           or if the particular operator doesn't apply (may happen with the XOR operator).

           For that we check that the sum of the heuristic is larger than zero which will avoid division by 0 error,
           although unlikely the situation may also occur with the pheromone trail if the random value generation yields a zero.
        '''
        heuristics = np.power(heuristics, self.beta)
        if heuristics.sum() == 0:
            heuristics += 1
        pheromones = np.power(pheromones, self.alpha)
        if pheromones.sum() == 0:
            pheromones += 1
        probabilities = pheromones*heuristics
        s = probabilities.sum()
        if s == 0:
            return np.ones(probabilities.shape)/probabilities.shape[0]
        return probabilities/s

    def compute_neighbour_sufs(self, neighbour, transformed_features,constructors, selected_node, current_su, X, y):
        '''Dynamical computation of the SU for the feature subset based on the adapted MIFS for the symmetrical uncertainty'''
        operator = neighbour[2]  # Get operator
        operands = [selected_node, neighbour[1]]  # Get operands (index,value)
        feature = create_feature(operator, operands)
        return self.compute_sufs_cached(current_su, transformed_features, feature.transform(X), constructors, feature, y, minimum=0)


        # return current_su + symmetrical_uncertainty(feature.transform(X),y)

    def set_cache(self, cache_loo, cache_heuristic):
        self.cache_loo = cache_loo
        self.cache_heuristic = cache_heuristic

    def compute_sufs_cached(self,current_su, transformed_features, transformed_feature, constructors, constructor, y, minimum=0):
        hashed_features = hash_features(constructors + [constructor])
        if hashed_features not in self.cache_heuristic:
            self.cache_heuristic[hashed_features] = compute_sufs(current_su, transformed_features, transformed_feature, y, minimum=0)
        return self.cache_heuristic[hashed_features]



    def evaluate_loo(self, features, classifier, transformed_features, y):
        hashed_features = hash_features(features)
        if hashed_features not in self.cache_loo:
            self.cache_loo[hashed_features] = classifier.leave_one_out_cross_val(
                transformed_features, y, fit=False)
        return self.cache_loo[hashed_features]

    def explore(self, X, y, graph, random_generator, parallel, max_errors=0):
        # np.random.seed(200)
        '''
        Search method that follows the following steps:
            1. The initial node is connected to all the others (roulette wheel selection is performed)
            2. There are 2 type of nodes (corresponding to an original feature (2.1) or corresponding to a value of a feature (2.2)):
                2.1. If the selected node is an original feature we add it to the selected subset and go to step 3.
                2.2. If the selected node is part of a logical feature then we select another node (the CONSTRUCTION step will not return full original features)
            3. Compute the score
                3.1. If it improves the previous one
                    3.1.1 Add the feature to the current subset
                    3.1.2 Update the score
                    3.1.3 Select another node (SELECTION step) 
                    3.1.4 Go to step 2
                3.2. If not, the exploration ends

        Note: Threading does not speed up the calculations as they are CPU bound and in python only I/O operations will benefit from this parallelism
              GPU improvement would reduce the time of the exploration.
        '''
        self.step = math.ceil(math.log2(X.shape[1]))
        self.current_features = []
        selected_nodes = set()
        constructed_nodes = set()
        classifier = NaiveBayes(encode_data=False, metric=self.metric)
        current_score = np.NINF
        score = 0
        if self.use_initials:
            self.current_features = [DummyFeatureConstructor(j) for j in range(X.shape[1])]
            classifier.fit(X, y)
            current_transformed_features_numpy = np.concatenate([f.transform(X) for f in self.current_features],axis=1)
            score = self.evaluate_loo(self.current_features, classifier, current_transformed_features_numpy, y)
            current_score = score
            selected_nodes.update(graph.get_original_ids())
        if len(self.current_features) == 0 :
            current_transformed_features_numpy = None
        
        initial, pheromones, heuristics = graph.get_initial_nodes(
            selected_nodes)

        probabilities = self.compute_probability(pheromones, heuristics)
        index = self.choose_next(probabilities, random_generator)
        node_id, selected_node = initial[index]



        # SU variable contains the MIFS-SU for the selected variable
        current_su = 0
        su = heuristics[index]

        is_fitted = self.use_initials
        feature_constructor = None
        n_errors = 0
        number_steps = 1
        while True:
            current_score = score
            if selected_node[1] is None:
                # Original Feature
                feature_constructor = DummyFeatureConstructor(selected_node[0])
                selected_nodes.add(node_id)
            else:
                # Need to construct next feature and compute heuristic value for the feature to replace temporal su from half-var
                neighbours, pheromones = graph.get_neighbours(
                    selected_node, constructed_nodes, step="CONSTRUCTION")
                # Compute heuristic
                # if not isinstance(self,FinalAnt):
                #     indexes = np.random.choice(np.arange(0,len(neighbours)),min(5,len(neighbours)),replace=False,p=pheromones/pheromones.sum())
                #     neighbours  = [ neighbours[index] for index in indexes]
                #     pheromones  = pheromones[indexes]
            

                if len(neighbours) == 0:
                    break
                # neighbours, pheromones = list(zip(*random.sample(list(zip(neighbours,pheromones)),math.ceil(len(neighbours)*0.25))))
                if self.beta != 0:
                    if parallel:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = []
                            for neighbour in neighbours:
                                futures.append(
                                    executor.submit(
                                        self.compute_neighbour_sufs, neighbour=neighbour,
                                        transformed_features = current_transformed_features_numpy,
                                        constructors = self.current_features,
                                        selected_node=selected_node,
                                        current_su=current_su,
                                        X=X, y=y))
                            concurrent.futures.wait(
                                futures, timeout=None, return_when='ALL_COMPLETED')
                            su = [future.result() for future in futures]
                    else:
                            su = [self.compute_neighbour_sufs( 
                                            neighbour=neighbour,
                                            transformed_features = current_transformed_features_numpy,
                                            selected_node=selected_node,
                                            constructors = self.current_features,
                                            current_su=current_su,
                                            X=X, y=y) for neighbour in neighbours]
                else:
                    su = np.ones(len(neighbours))

                probabilities = self.compute_probability(
                    pheromones, np.array(su))
                index = self.choose_next(probabilities, random_generator)

                su = su[index]
                feature_constructor = create_feature(
                    neighbours[index][2], [selected_node, neighbours[index][1]])
                constructed_nodes.add(
                    frozenset((node_id, neighbours[index][0], neighbours[index][2])))
                node_id, selected_node = neighbours[index][:2]

            # Assess new feature
            transformed_feature = feature_constructor.transform(X)
            if is_fitted:
                classifier.add_features(transformed_feature, y)
            else:
                classifier.fit(transformed_feature, y)
                is_fitted = True
            if current_transformed_features_numpy is None:
                current_transformed_features_numpy = transformed_feature
            else:
                current_transformed_features_numpy = append_column_to_numpy(current_transformed_features_numpy,transformed_feature)
            if number_steps >= self.step:
                score = self.evaluate_loo(
                    self.current_features+[feature_constructor], classifier, current_transformed_features_numpy, y)
                if score <= current_score:
                    if n_errors >= max_errors:
                        break
                    else:
                        n_errors += 1
                else:
                    n_errors = 0
                number_steps = 0
            else:
                number_steps +=1
            current_su = su
            self.current_features.append(feature_constructor)
            current_score = score
            # Select next
            neighbours, pheromones = graph.get_neighbours(
                selected_node, selected_nodes, step="SELECTION")
            
            # Compute heuristic
            su = []
            # if len(neighbours)==0:
            #     break
            if self.beta != 0:
                for neighbour,pheromone in  zip(neighbours, pheromones):
                    if neighbour[1][1] is None:
                        # Original variable
                        su.append(self.compute_sufs_cached(current_su, current_transformed_features_numpy, X[:, neighbour[1][0]],self.current_features,DummyFeatureConstructor(neighbour[1][0]), y, minimum=0))
                    else:
                        # This is a temporal variable that will not be finally selected but only used to calculate the heuristic
                        su.append(self.compute_sufs_cached(current_su,current_transformed_features_numpy,X[:, neighbour[1][0]] == neighbour[1][1],self.current_features,FeatureOperand(feature_index = neighbour[1][0], value = neighbour[1][1]) ,y,minimum=0))
                        # su.append(1)
                        #Look two steps ahead
                        # neighbours_next, _ = graph.get_neighbours(
                        #     neighbour[1], constructed_nodes, step="CONSTRUCTION")
                        # su.append(max(self.compute_neighbour_sufs(
                        #                 neighbour=neigbour_next,
                        #                 transformed_features = current_transformed_features_numpy,
                        #                 constructors = self.current_features,
                        #                 selected_node=selected_node,
                        #                 current_su=current_su,
                        #                 X=X, y=y)
                        #     for neigbour_next in neighbours_next))
                        # su.append(compute_sufs(current_su,[f.transform(X).flatten() for f in self.current_features],X[:, neighbour[1][0]] == neighbour[1][1],y,minimum=0))
            else:
                su = np.ones(len(neighbours))
            probabilities = self.compute_probability(pheromones, np.array(su))
            index = self.choose_next(probabilities, random_generator)

            su = su[index]
            node_id, selected_node = neighbours[index][:2]
        if current_transformed_features_numpy.shape[1] > len(self.current_features):
            current_transformed_features_numpy = np.delete(current_transformed_features_numpy,-1,axis=1)
        self.final_score = self.evaluate_loo(self.current_features, classifier, current_transformed_features_numpy, y)
                
        return self.final_score

    def run(self, X, y, graph, random_generator, parallel=False, max_errors=0):
        # print(f"Ant [{self.ant_id}] running in thread [{threading.get_ident()}]")
        return self.explore(X, y, graph, random_generator, parallel, max_errors)


class FinalAnt(Ant):
    def choose_next(self, probabilities, random_generator):
        '''Selects index based on roulette wheel selection'''
        return np.argmax(probabilities)