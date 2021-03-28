import concurrent
import random
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tqdm.autonotebook import tqdm

from tfg.ant_colony import AntFeatureGraph
from tfg.ant_colony import AntFeatureGraphMI
from tfg.ant_colony import Ant
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.naive_bayes import NaiveBayes
from tfg.utils import translate_features

class ACFCS(TransformerMixin,ClassifierMixin,BaseEstimator):
    def __init__(self,
                ants=10, 
                evaporation_rate=0.05,
                intensification_factor=0.05,
                alpha=1.0, 
                beta=0.0, 
                beta_evaporation_rate=0.05,
                iterations=100, 
                early_stopping=20,
                seed = None,
                parallel=False,
                save_features=True,
                path=None,
                filename=None,
                verbose=0,
                graph_strategy = "mutual_info",
                connections = 2):
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

        allowed_graph_strategy = ("full","mutual_info")
        if self.graph_strategy not in allowed_graph_strategy:
            raise ValueError("Unknown graph strategy type: %s, expected one of %s." % (self.graph_strategy, allowed_graph_strategy))

    def fit(self,X,y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()

        self.categories_ = None
        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        X = self.feature_encoder_.fit_transform(X)
        y = self.class_encoder_.fit_transform(y)

        if self.graph_strategy=="full":
                self.afg = AntFeatureGraph(seed=self.seed).compute_graph(X, y, ("XOR","OR", "AND"))
        else:
                self.afg = AntFeatureGraphMI(seed=self.seed,connections=self.connections).compute_graph(X, y, ("XOR","OR", "AND"))
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
                                    "p_matrix_c": len(self.afg.pheromone_matrix_attribute_completion),
                                    "p_matrix_s": len(self.afg.pheromone_matrix_selection),
                                    "distance_from_best": distance_from_best})
            ants = [Ant(ant_id=i,alpha=self.alpha,beta=beta) for i in range(self.ants)]
            beta*=self.beta_evaporation_rate
            results = []
            for ant in ants:
                    results.append(ant.run(X=X,y=y,graph=self.afg,random_generator=random,parallel=self.parallel))
            results = np.array(results)
            best_ant = np.argmax(results)
            distance_from_best = np.mean(np.abs(results-best_score))
            self.afg.update_pheromone_matrix_evaporation(self.evaporation_rate)
            if results[best_ant] > best_score:
                ant = ants[best_ant]
                iterations_without_improvement = 0
                best_score = results[best_ant]
                self.best_features = ant.current_features
                self.afg.intensify(self.best_features,self.intensification_factor)
            else:
                iterations_without_improvement+=1
                if iterations_without_improvement > self.early_stopping:
                    break
            
        if self.save_features:
            translate_features(features=self.best_features,
                                feature_encoder = self.feature_encoder_,
                                categories=self.categories_,
                                path=self.path,
                                filename=self.filename)


        self.classifier_ = NaiveBayes(encode_data=False)
        self.classifier_.fit(np.concatenate([ f.transform(X) for f in self.best_features],axis=1),y)
        self.backwards_fss(X,y)
        return self

    def backwards_fss(self,X,y):
        check_is_fitted(self)
        improvement = True
        best_features = np.concatenate([ f.transform(X) for f in self.best_features],axis=1)
        best_score = self.classifier_.leave_one_out_cross_val(best_features,y,fit=False)
        while improvement and best_features.shape[1] >1:
            improvement = False
            feature = None
            for i in range(best_features.shape[1]):
                feature = best_features[:,i].reshape(-1,1)
                current_features = np.delete(best_features,i,axis=1)
                self.classifier_.remove_feature(i)
                current_score = self.classifier_.leave_one_out_cross_val(current_features,y,fit=False)
                self.classifier_.add_features(feature,y,[i])
                if current_score > best_score:
                    feature_index = i
                    improvement = True
                    best_score = current_score
                
            if improvement:
                best_features = np.delete(best_features,feature_index,axis=1)
                self.classifier_.remove_feature(feature_index)
                del self.best_features[feature_index]
        return
                
               
    def transform(self,X,y):
        check_is_fitted(self)
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        X = self.feature_encoder_.transform(X)
        y = self.class_encoder_.transform(y)
        X = np.concatenate([ f.transform(X) for f in self.best_features],axis=1)
        return X,y

    def predict(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier_.predict(X,y)

        
    def predict_proba(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier_.predict_proba(X,y)

    def score(self,X,y):
        X,y = self.transform(X,y)
        return self.classifier_.score(X,y)
 