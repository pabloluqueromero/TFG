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

class GeneticAlgorithm(TransformerMixin,ClassifierMixin,BaseEstimator):
    def evaluate(self, individual, X, y):
        classifier = NaiveBayes(encode_data=False)
        return classifier.leave_one_out_cross_val(transform_features(individual, X), y, fit=True)

    def fitness(self, population,X,y):
        evaluation = []
        for individual in population:
            if self.use_initials:
                evaluation.append((individual, self.evaluate(self.backward_features + individual,X,y)))
            else:
                evaluation.append((individual, self.evaluate(individual,X,y)))
        return evaluation

    def generate_population(self):
        population = []
        for _ in range(self.individuals):
            individual = []
            for _ in range(self.size):
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
                individual.append(create_feature(operator=op, operands=operands))
            population.append(individual)
        return population

    def mutate(self, population):
        new_population = []
        for individual in population:
            if random.random() < self.mutation_probability:
                number_of_chromosomes = random.sample(list(range(len(individual))),random.randint(1,self.size-1))
                for i in number_of_chromosomes:
                    feature = individual[i]
                    feature.op = random.choice(('OR','XOR','AND'))
                    for operand in feature.operands:
                        operand.feature_index = random.randint(0,self.n_features-1)
                        operand.value = random.randint(0,self.unique_values[operand.feature_index]-1)
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
            for individual in population:
                cumulative_prob += individual[1]/totalFitness
                if r <= cumulative_prob:
                    selected_individuals.append(individual[0].copy())
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
            for i, individual in enumerate(population, start=1):
                cumulative_prob += (num_selected-i+1)/totalRank
                if r <= cumulative_prob:
                    selected_individuals.append(individual[0].copy())
                    break
        return selected_individuals

    def fit(self,X,y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()

        if isinstance(X,pd.DataFrame):
            self.categories_ = X.columns
        X = self.feature_encoder_.fit_transform(X)
        y = self.class_encoder_.fit_transform(y)
        
        classifier = NaiveBayes(encode_data = False,n_intervals=self.n_intervals,metric=self.metric)
        if self.use_initials:
            classifier.fit(X,y)
            current_features = [DummyFeatureConstructor(j) for j in range(X.shape[1])]
            self.backward_features = backward_search(X,y,current_features,classifier)
        self.n_features = X.shape[1]
        self.unique_values = [values.shape[0] for values in self.feature_encoder_.categories_]
        random.seed(self.seed)

        best_individual = self.execute_algorithm(X,y)
        self.best_features = best_individual
        self.classifier_ = NaiveBayes(encode_data=False)
        self.classifier_.fit(self.transform(X,y)[0],y)

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
        if self.use_initials:
            return self.backward_features+max(population_with_fitness,key=lambda x: x[1])[0]
        return list(max(population_with_fitness,key=lambda x: x[1])[0])

    def __init__(self, 
                 size=10, 
                 seed=None, 
                 individuals=1, 
                 generations=40,
                 mutation_probability=0.2, 
                 selection="rank", 
                 combine="truncation",
                 n_intervals = 5,
                 metric = "accuracy",
                 use_initials=True,
                 verbose=False
                 ):
        self.size = size
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
