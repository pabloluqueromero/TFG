import concurrent
import random
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from tqdm.autonotebook import tqdm

from tfg.ant_colony import AntFeatureGraph
from tfg.ant_colony import AntFeatureGraphMI
from tfg.ant_colony import Ant
from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.utils import translate_features

class ACFCS(ClassifierMixin,BaseEstimator):
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
                filename=None):
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

    def fit(self,X,y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.label_encoder_ = LabelEncoder()

        self.categories_ = None
        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        X = self.feature_encoder_.fit_transform(X)
        y = self.label_encoder_.fit_transform(y)

        # self.afg = AntFeatureGraph(seed=self.seed).compute_graph(X, y, ("XOR","OR", "AND"))
        self.afg = AntFeatureGraphMI(seed=self.seed).compute_graph(X, y, ("XOR","OR", "AND"))
        print(f"Number of nodes: {len(self.afg.nodes)}")
        random.seed(self.seed)
        best_score = 0
        best_features = []
        iterations_without_improvement = 0
        iterator = tqdm(range(self.iterations))
        beta = self.beta
        distance_from_best = -1
        for iteration in iterator:
            iterator.set_postfix({"best_score":best_score,
                                  "n_features":len(best_features),
                                  "p_matrix_c": len(self.afg.pheromone_matrix_attribute_completion),
                                  "p_matrix_s": len(self.afg.pheromone_matrix_selection),
                                  "distance_from_best": distance_from_best
                                  })
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
                best_features = ant.current_features
                self.afg.intensify(best_features,self.intensification_factor)
            else:
                iterations_without_improvement+=1
                if iterations_without_improvement > self.early_stopping:
                    break
            
        if self.save_features:
            translate_features(features=best_features,
                                feature_encoder = self.feature_encoder_,
                                categories=self.categories_,
                                path=self.path,
                                filename=self.filename)
        return self