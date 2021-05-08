import random
from tfg.naive_bayes import NaiveBayes
from tfg.utils import backward_search, transform_features
from tfg.feature_construction import DummyFeatureConstructor, create_feature
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder

def memoize(f):
    cache = dict()
    def g(individual, X, y):
        hashable_individual = tuple(individual[0]),tuple(individual[1]),frozenset(individual[1])
        hash_individual = hash(hashable_individual)
        if hash_individual not in cache:
            cache[hash_individual] = f(individual, X, y)
        return cache[hash_individual]
    return g
class GeneticAlgorithmV2(TransformerMixin,ClassifierMixin,BaseEstimator):
    def simple_evaluate(self, individual, X, y):
        classifier = NaiveBayes(encode_data=False)
        return classifier.leave_one_out_cross_val(transform_features(individual[0]+individual[1], X), y, fit=True)

    def fitness(self, population,X,y):
        evaluation = []
        for individual in population:
            evaluation.append((individual, self.evaluate(individual,X,y)))
        return evaluation

    def generate_population(self):
        population = []
        for _ in range(self.individuals):
            individual = ([],[],set())
            if self.flexible_logic:
                n_chromosomes = range(random.randint(1,self.size))
            else:
                n_chromosomes = range(self.size)

            for _ in n_chromosomes:
                operand1_feature = random.randint(0, self.n_features-1)
                operand2_feature = random.randint(0, self.n_features-1)
                if operand1_feature == operand2_feature:
                    op = 'OR'
                    operand1_value = random.randint(0,self.unique_values[operand1_feature]-1) 
                    operand2_value = random.randint(0,self.unique_values[operand1_feature]-1)
                else:
                    op = random.choice(('OR', 'XOR', 'AND'))
                    operand1_value = random.randint(0, self.unique_values[operand1_feature]-1)
                    operand2_value = random.randint(0, self.unique_values[operand2_feature]-1)
                operands  = []
                operands.append((operand1_feature,operand1_value))
                operands.append((operand2_feature,operand2_value))
                individual[1].append(create_feature(operator=op, operands=operands))
            n_og_features = random.randint(0,self.n_features-1)
            features = list(range(self.n_features))
            for f in random.sample(features,n_og_features):
                individual[0].append(DummyFeatureConstructor(feature_index=f))
                individual[2].add(f)
            population.append(individual)
        return population

    def mutate(self, population):
        new_population = []
        for individual in population:
            if random.random() < self.mutation_probability:
                chromosomes_index = []
                if self.flexible_logic:
                    if len(individual[1])>0:
                        chromosomes_index = random.sample(list(range(len(individual[1]))),random.randint(1,len(individual[1])))
                    else:
                        op = random.choice(('OR','XOR','AND'))
                        operands = []
                        for _ in range(2):
                            feature_index = random.randint(0,self.n_features-1)
                            value = random.randint(0,self.unique_values[feature_index]-1)
                            operands.append((feature_index,value))
                        individual[1].append(create_feature(operator=op, operands=operands))
                        continue

                else:
                    chromosomes_index = random.sample(list(range(len(individual[1]))),random.randint(1,len(individual[1])))


                for i in range(len(chromosomes_index)):
                    index = chromosomes_index[i]
                    if not self.flexible_logic:
                        feature = individual[1][index]
                        feature.op = random.choice(('OR','XOR','AND'))
                        for operand in feature.operands:
                            operand.feature_index = random.randint(0,self.n_features-1)
                            operand.value = random.randint(0,self.unique_values[operand.feature_index]-1)
                    else:
                        a =  random.random() 
                        if a <0.33:
                            feature = individual[1][index]
                            feature.op = random.choice(('OR','XOR','AND'))
                            for operand in feature.operands:
                                operand.feature_index = random.randint(0,self.n_features-1)
                                operand.value = random.randint(0,self.unique_values[operand.feature_index]-1)
                        elif a<0.66:
                            op = random.choice(('OR','XOR','AND'))
                            operands = []
                            for _ in range(2):
                                feature_index = random.randint(0,self.n_features-1)
                                value = random.randint(0,self.unique_values[feature_index]-1)
                                operands.append((feature_index,value))
                            individual[1].append(create_feature(operator=op, operands=operands))

                        else:
                            del individual[1][index]
                            chromosomes_index = [ j-1 if j > index else j for j in chromosomes_index]

            if random.random() < self.mutation_probability:
                a =  random.random() 
                og_features = individual[0]
                included_features = individual[2]
                if (a <0.33 and len(og_features) < self.n_features) or len(og_features)==0:
                    selected = random.choice(tuple(set(list(range(0,self.n_features))) - included_features))
                    included_features.add(selected)
                    og_features.append(DummyFeatureConstructor(selected))
                elif a<0.66 and len(og_features) < self.n_features and len(og_features)>0:
                    selected = random.choice(tuple(set(list(range(0,self.n_features))) - included_features))
                    index = random.randint(0,len(og_features)-1)
                    feature = og_features[index].feature_index
                    og_features[index] = DummyFeatureConstructor(selected)
                    included_features.remove(feature)
                    included_features.add(selected)
                else:
                    index = random.randint(0,len(og_features)-1)
                    feature = og_features[index].feature_index
                    del og_features[index]
                    included_features.remove(feature)
            new_population.append(individual)
        return new_population

    def elitism(self, population1, population2):
        maximum = max(population2, key=lambda x: x[1])
        minimum_index = min(enumerate(population1),
                            key=lambda x: x[1][1])[0]
        population1[minimum_index] = maximum
        return population1

    def truncation(self, population1, population2):
        return sorted(population1 + population2, reverse=True, key=lambda x: x[1])[:len(population1)]

    # def crossover(self, population):
    #     new_population = []
    #     possible_crossover_points = range(self.size+1, self.size**2-self.size)
    #     minimum = min(self.number_points, self.size*(self.size-2)-1)
    #     last = self.size**2-self.size
    #     for individual1, individual2 in zip(population, population[1:]):
    #         if random.random() < self.crossover_probability:
    #             i1 = individual1[0]
    #             i2 = individual2[0]
    #             elements = [
    #                 self.size]+sorted(random.sample(possible_crossover_points, minimum))+[last]
    #             offspring1 = set()
    #             offspring2 = set()

    #             for point in range(len(elements) - 1):
    #                 section = set(range(elements[point], elements[point+1]))
    #                 offspring1 |= section.intersection(i1)
    #                 offspring2 |= section.intersection(i2)
    #                 i1, i2 = i2, i1

    #             new_population.append(frozenset(offspring1))
    #             new_population.append(frozenset(offspring2))

    #         else:
    #             new_population.append(individual1[0])
    #             new_population.append(individual2[0])
    #     return new_population

    def select_population(self, population):
        selected_individuals = []
        num_selected = len(population)
        totalFitness = sum(fitness
                           for _, fitness in population)
        for _ in range(num_selected):
            cumulative_prob = 0.0
            r = random.random()
            for individual_with_fitness in population:
                cumulative_prob += individual_with_fitness[1]/totalFitness
                if r <= cumulative_prob:
                    selected_individuals.append(self.copy_individual(individual_with_fitness[0]))
                    break
        return selected_individuals

    def select_population_rank(self, population):
        selected_individuals = []
        num_selected = len(population)
        totalRank = (num_selected * (num_selected+1))/2
        population.sort(reverse=True, key=lambda x: x[1])
        for _ in range(num_selected):
            cumulative_prob = 0.0
            r = random.random()
            for i, individual_with_fitness in enumerate(population, start=1):
                cumulative_prob += (num_selected-i+1)/totalRank
                if r <= cumulative_prob:
                    selected_individuals.append(self.copy_individual(individual_with_fitness[0]))
                    break
        return selected_individuals

    def copy_individual(self,individual):
        return ([chrms.copy() for chrms in individual[0]],[chrms.copy() for chrms in individual[1]],individual[2].copy())

    def fit(self,X,y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()

        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        if self.encode:
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)
        
        classifier = NaiveBayes(encode_data = False,n_intervals=self.n_intervals,metric=self.metric)
        self.n_features = X.shape[1]
        if self.encode:
            self.unique_values = [values.shape[0] for values in self.feature_encoder_.categories_]
        else:
            self.unique_values = [np.unique(X[:,j]).shape[0] for j in range(X.shape[1])]
        random.seed(self.seed)

        best_individual = self.execute_algorithm(X,y)
        self.best_features = best_individual
        self.classifier_ = NaiveBayes(encode_data=False)
        self.classifier_.fit(self.transform(X,y)[0],y)
        self.best_features = backward_search(X,y,self.best_features,self.classifier_)

    def execute_algorithm(self,X,y):
        population = self.generate_population()
        population_with_fitness = self.fitness(population,X,y)        
        iterator = tqdm(range(self.generations), leave=False) if self.verbose else range(self.generations)
        for generation in iterator:
            selected_individuals = self.selection(population_with_fitness)
            crossed_individuals = selected_individuals#self.crossover(selected_individuals)
            mutated_individuals = self.mutate(crossed_individuals)
            new_population = self.fitness(mutated_individuals,X,y)
            population_with_fitness = self.combine(
                population_with_fitness, new_population)

            # Obtaining population's statistics
            if self.verbose:
                best, mean = get_max_mean(population_with_fitness)
                iterator.set_postfix({"Generation":generation,
                                    "populationLength":len(population_with_fitness),
                                    "best fitness": best,
                                    "mean fitness": mean})

        # print("REPEATED INDIVIDUALS:", self.number_individuals *
        #       2*self.generations-self.not_repeated)
        best_individual= max(population_with_fitness,key=lambda x: x[1])[0]
        return best_individual[0]+best_individual[1]
    
    def reset_evaluation(self):
        self.evaluate = memoize(self.simple_evaluate)

    def __init__(self, 
                 size=10, 
                 seed=None, 
                 individuals=1, 
                 generations=40,
                 mutation_probability=0.2, 
                 selection="rank", 
                 combine="elitism",
                 n_intervals = 5,
                 metric = "accuracy",
                 use_initials=True,
                 flexible_logic = True,
                 verbose=False,
                 encode=True
                 ):
        self.size = size
        self.encode=encode
        self.flexible_logic = flexible_logic
        self.verbose = verbose
        self.n_intervals = n_intervals
        self.metric = metric
        self.seed = seed
        self.individuals = individuals
        self.generations = generations
        self.use_initials = use_initials
        # self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.selection = self.select_population_rank if "rank" in selection else self.select_population
        self.combine = self.elitism if "elit" in combine else self.truncation
        self.reset_evaluation()
    
    def set_params(self, **params):
        super().set_params(**params)
        if "selection" in params: 
            self.selection = self.select_population_rank if "rank" in params["selection"] else self.select_population
        if "combine" in params: 
            self.combine = self.elitism if "elit" in params["combine"] else self.truncation
        
    def transform(self,X,y):
        check_is_fitted(self)
        if isinstance(y,pd.DataFrame):
            y = y.to_numpy()
        if self.encode:
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
 


def get_max_mean(population_with_fitness):
    best_score = -1
    cumul = 0
    for individual,fitness in population_with_fitness:
        cumul+=fitness
        best_score = max(best_score,fitness)
    return best_score, cumul/len(population_with_fitness)
