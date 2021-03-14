import concurrent
import numpy as np
import random


from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from tqdm.autonotebook import tqdm

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.ant_colony import AntFeatureGraph
from tfg.ant_colony import Ant

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
                parallel=False):
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

    def fit(self,X,y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.label_encoder_ = LabelEncoder()

        self.categories_ = None
        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        X = self.feature_encoder_.fit_transform(X)
        y = self.label_encoder_.fit_transform(y)

        self.afg = AntFeatureGraph(seed=self.seed).compute_graph(X, y, ("XOR","OR", "AND"))
        print(f"Number of nodes: {len(self.afg.nodes)}")
        random.seed(self.seed)
        best_score = 0
        best_features = []
        iterations_without_improvement = 0
        iterator = tqdm(range(self.iterations))
        iteration_n=0
        beta = self.beta
        distance_from_best = -1
        for iteration in iterator:
            iteration_n+=1
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
            
            
        translate_features(features=best_features,
                            feature_encoder = self.feature_encoder_,
                            categories=self.categories_,
                            path="./features_translated")
        return self





######################################################################
#
#
#                               TESTS -> to be removed   
#               
#
######################################################################
# import pandas as pd
# import os
# from tfg.utils._utils import translate_features
# def get_X_y(base_path, name, data, test, label):
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
#     X = df.drop([label], axis=1)
#     y = df[label]
#     return X, y


# # base_path, data = "../Dataset/UCIREPO/", ["anneal", "label"]
# base_path, data = "../Dataset/UCIREPO/", ["wisconsin", "diagnosis"]
# # base_path, data = "../Dataset/UCIREPO/", ["iris", "Species"]

# X, y = get_X_y(base_path, data[0], data[0] +
#                ".data.csv", test="as", label=data[1])

# # feature_encoder_ = CustomOrdinalFeatureEncoder()
# # label_encoder_ = LabelEncoder()
# # X = feature_encoder_.fit_transform(X)
# # y = label_encoder_.fit_transform(y)
# # afg = AntFeatureGraph(seed=2).compute_graph(X, y, ("OR", "AND"))
# # neighbours = afg.get_neighbours((1, 1), "CONSTRUCTION")
# # print(neighbours)
# from tfg.naive_bayes import NaiveBayes
# # print(X[["SkinThickness"]].shape)
# # print(NaiveBayes(encode_data=True).leave_one_out_cross_val(X[["BloodPressure"]],y,fit=True))
# from time import time
# from tfg.ranker import RankerLogicalFeatureConstructor
# params = {"strategy":"eager","block_size":2,"verbose":1}
# # st=time()
# # r = RankerLogicalFeatureConstructor(**params).fit(X,y)
# # print(time()-st)
# st = time()
# # ACFCS(ants=10, 
# #     evaporation_rate=0.5,
# #     intensification_factor=0.05,
# #     alpha=0.5, 
# #     beta=0.5, 
# #     beta_evaporation_rate=0.05,
# #     iterations=100, 
# #     early_stopping=20,
# #     seed = 3).fit(X, y)
# ACFCS(ants=20, 
#     evaporation_rate=0.1,
#     intensification_factor=3,
#     alpha=1, 
#     beta=0.9, 
#     beta_evaporation_rate=0.1,
#     iterations=100, 
#     early_stopping=20,
#     seed = 3,
#     parallel=False).fit(X, y)
# print(time()-st)
# # afg.update_pheromone_matrix_evaporation(0.05)
