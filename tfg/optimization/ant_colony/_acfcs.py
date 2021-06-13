import concurrent
import random
import numpy as np
import pandas as pd

#Sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#Auxiliary imports
from tqdm.autonotebook import tqdm

#TFG imporst
from tfg.optimization.ant_colony import AntFeatureGraph
from tfg.optimization.ant_colony import AntFeatureGraphMI
from tfg.optimization.ant_colony import Ant, FinalAnt
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import create_feature,DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.utils import translate_features,append_column_to_numpy

class ACFCS(TransformerMixin,ClassifierMixin,BaseEstimator):
    def __init__(self,
                ants=10, 
                evaporation_rate=0.05,
                intensification_factor=0.05,
                alpha=1.0, 
                beta=0.0, 
                beta_evaporation_rate=0.05,
                step = 1,
                iterations=100, 
                early_stopping=20,
                update_strategy="best",
                seed = None,
                parallel=False,
                save_features=False,
                path=None,
                filename=None,
                verbose=0,
                graph_strategy = "mutual_info",
                connections = 2,
                max_errors=0,
                metric="accuracy",
                use_initials=False,
                final_selection="ALL",
                encode_data=True,
                backwards=False):
        self.backwards = backwards
        self.step = step
        self.backwards = backwards
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.intensification_factor = intensification_factor
        self.alpha = alpha
        self.beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.iterations = iterations
        self.early_stopping = early_stopping
        self.seed = seed
        self.parallel = parallel
        self.save_features = save_features
        self.path = path
        self.filename = filename
        self.verbose=verbose
        self.graph_strategy = graph_strategy
        self.connections = connections
        self.metric = metric
        self.update_strategy = update_strategy
        self.use_initials = use_initials
        self.final_selection = final_selection
        self.encode_data = encode_data
        self.max_errors = max_errors
        allowed_graph_strategy = ("full","mutual_info")
        if self.graph_strategy not in allowed_graph_strategy:
            raise ValueError("Unknown graph strategy type: %s, expected one of %s." % (self.graph_strategy, allowed_graph_strategy))
        
        allowed_update_strategy = ("all","best")
        if self.update_strategy not in allowed_update_strategy:
            raise ValueError("Unknown graph strategy type: %s, expected one of %s." % (self.update_strategy, allowed_update_strategy))

        self.reset_cache()

    def reset_cache(self):
        self.cache_loo = dict()
        self.cache_heuristic = dict()

    def fit(self,X,y,init_graph=True):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()

        self.categories_ = None
        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        if self.encode_data:
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)
        if init_graph:
            if self.graph_strategy=="full":
                    self.afg = AntFeatureGraph(seed=self.seed).compute_graph(X, y, ("XOR","OR", "AND"))
            else:
                    self.afg = AntFeatureGraphMI(seed=self.seed,connections=self.connections).compute_graph(X, y, ("XOR","OR", "AND"))
        else:
            self.afg.reset_pheromones()
        if self.verbose:
            print(f"Number of nodes: {len(self.afg.nodes)}")
        random.seed(self.seed)
        best_score = 0
        self.best_features = []
        iterations_without_improvement = 0
        iterator = tqdm(range(self.iterations)) if self.verbose else range(self.iterations)
        beta = self.beta
        distance_from_best = -1
        for iteration in iterator:
            if self.verbose:
                iterator.set_postfix({"best_score":best_score,
                                    "n_features":len(self.best_features),
                                    "p_matrix_c": len(self.afg.pheromone_construction),
                                    "p_matrix_s": len(self.afg.pheromone_selection),
                                    "distance_from_best": distance_from_best})
            ants = [Ant(ant_id=i,alpha=self.alpha,beta=beta, metric = self.metric, use_initials = self.use_initials, cache_loo = self.cache_loo, cache_heuristic = self.cache_heuristic,step = self.step) for i in range(self.ants)]
            beta*=(1-self.beta_evaporation_rate)
            results = []
            for ant in ants:
                results.append(ant.run(X=X,y=y,graph=self.afg,random_generator=random,parallel=self.parallel,max_errors = self.max_errors))
            results = np.array(results)
            
            self.afg.update_pheromone_matrix_evaporation(self.evaporation_rate)
            distance_from_best = np.mean(np.abs(results-best_score))
            best_ant = np.argmax(results)
            if self.update_strategy == "best":
                ant = ants[best_ant]
                self.afg.intensify(ant.current_features,self.intensification_factor,1,self.use_initials)
            else:
                for ant_score,ant in zip(results,ants):
                    self.afg.intensify(ant.current_features,self.intensification_factor,ant_score,self.use_initials)


            if results[best_ant] >= best_score:
                iterations_without_improvement = 0
                ant = ants[best_ant]
                best_score = results[best_ant]
                self.best_features = ant.current_features
            else:
                iterations_without_improvement+=1
                if iterations_without_improvement > self.early_stopping:
                    break


        self.classifier_ = NaiveBayes(encode_data=False,metric = self.metric)
        # self.best_features = self.get_best_features(self.afg,X,y)
        if self.final_selection=="BEST":
            # self.best_features = self.get_best_features(self.afg,X,y)
            pass
        else:
            final_ant = FinalAnt(ant_id=0,alpha=self.alpha,beta=beta, metric = self.metric,use_initials = self.use_initials, cache_loo = self.cache_loo, cache_heuristic = self.cache_heuristic,step = self.step)
            final_ant.run(X=X,y=y,graph=self.afg,random_generator=random,parallel=self.parallel)
            self.best_features = final_ant.current_features
        self.classifier_.fit(np.concatenate([feature.transform(X) for feature in self.best_features],axis=1),y)
        if self.backwards:
            self.backwards_fss(X,y)
            
        if self.save_features:
            translate_features(features=self.best_features,
                                feature_encoder = self.feature_encoder_,
                                categories=self.categories_,
                                path=self.path,
                                filename=self.filename)
        return self

    def get_best_features(self,afg,X,y):
        # def get_best_neighbours(self,pheromone,heuristic):
        #     return np.argmax(np.pow(pheromone,self.alpha)* np.pow(heuristic,self.beta))
        nodes,pheromones,_ = afg.get_initial_nodes(set())
        node_id,selected_node  = nodes[np.argmax(pheromones)]#get_best_neighbours(pheromones,heuristic)
        selected_nodes = set()
        constructed_nodes = set()
        classifier = NaiveBayes(encode_data=False,metric=self.metric)
        is_fitted = False
        current_features = []
        current_score = 0
        score=0
        while True:
            current_score = score
            if selected_node[1] is None:
                feature_constructor = DummyFeatureConstructor(selected_node[0])
                selected_nodes.add(node_id)
            else:
                # Need to construct next feature and compute heuristic value for the feature to remplace temporal su from half-var
                neighbours,pheromones = self.afg.get_neighbours(selected_node, constructed_nodes, step="CONSTRUCTION")
                # Compute heuristic
                if len(neighbours)==0:
                    break                
                index = np.argmax(pheromones)
                feature_constructor = create_feature(neighbours[index][2],[selected_node,neighbours[index][1]])
                constructed_nodes.add(frozenset((node_id,neighbours[index][0],neighbours[index][2])))
                node_id,selected_node = neighbours[index][:2]
            #Assess new feature
            transformed_feature = feature_constructor.transform(X)
            if is_fitted:
                classifier.add_features(transformed_feature, y)
            else:
                classifier.fit(transformed_feature, y)
                is_fitted = True
            features = append_column_to_numpy(features,transformed_feature)#np.concatenate([f.transform(X) for f in current_features]+[transformed_feature],axis=1)   
            score = classifier.leave_one_out_cross_val(features, y,fit=False)
            if score <= current_score:
                break
            current_features.append(feature_constructor)
        return current_features
 