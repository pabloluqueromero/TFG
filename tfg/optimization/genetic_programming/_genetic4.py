import numpy as np
import pandas as pd
import random

import GeneticUtils

from itertools import chain
from time import time
from tqdm.autonotebook import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor, create_feature
from tfg.naive_bayes import NaiveBayes
from tfg.optimization import OptimizationMixin
from tfg.utils import compute_sufs, compute_sufs_non_incremental, memoize_genetic, transform_features
from tfg.utils import get_max_mean


class GeneticProgrammingFlexibleLogic(OptimizationMixin, TransformerMixin, ClassifierMixin, BaseEstimator):
    """GeneticProgramming for Feature Construction and Selection.


    Parameters
    ----------

    seed : int or None
        Seed to guarantee reproducibility

    individuals : int
        Number of individuals per population

    generations : int
        Number of generations 

    mutation_probability : float
        Probability for each individual of being mutated

    select : {rank,proportionate}
        Selection strategy

    mutation : {simple,complex}
        Mutation strategy

    combine : {truncation,elitism} 
        Population combination strategy

    n_intervals : int
        Number of intervals for the discretization of continous variables

    mixed : bool
        Mix heuristic and wrapper evaluation

    mixed_percentage : float
        Percentage of total iterations to do heuristic evaluation

    metric : {accuracy,f1-score}
        Target metric for the optimization process
    
    flexible_logic: bool
        Allow different individual sizes in the generation
    
    encode_data : bool, default=True
        Encode data when data is not encoded by default with an OrdinalEncoder
    
    verbose :int {0,1}, default = 1 
        Display process progress


    Attributes
    ----------
    classifier_ : NaiveBayes
        Base classifier used for prediction

    best_features_ : array-lik of Feature
        Array of selected Feature used for transforming new data
    """

    def simple_evaluate(self, individual, X, y):
        classifier_ = NaiveBayes(encode_data=False, metric=self.metric)
        return classifier_.leave_one_out_cross_val(transform_features(individual[0]+individual[1], X), y, fit=True)

    def simple_evaluate_heuristic(self, individual, X, y):
        return compute_sufs_non_incremental(features=[f.transform(X) for f in chain(*individual[:2])], y=y)

    def fitness(self, population, X, y):
        return [(individual, self.evaluate(individual, X, y)) for individual in population]

    def generate_population(self):
        population = []
        utils = GeneticUtils.with_instance(self)
        for _ in range(self.individuals):
            individual = Individual()
            for _ in utils.get_inidividual_size():
                individual.add_constructed(utils.generate_feature(random))
                
            n_og_features = random.randint(0, self.n_features-1)
            features = list(range(self.n_features))
            for original_index in random.sample(features, n_og_features):
                individual.add_original(original_index)
            population.append(individual)
        return population

    def mutate_complex(self, population, **kwargs):
        for individual in population:
            if random.random() < self.mutation_probability:
                 GeneticUtils \
                        .with_instance(self) \
                        .complex_mutate_constructed_features(individual)
                
            if random.random() < self.mutation_probability:
                GeneticUtils \
                        .with_instance(self) \
                        .complex_mutate_original_features(individual)

            GeneticUtils \
                    .with_instance(self) \
                    .fix_individual(individual)

    def mutate_simple(self, population, **kwargs):
        new_population = []
        for individual in population:
            if random.random() < self.mutation_probability:
                 GeneticUtils \
                        .with_instance(self) \
                        .simple_mutate_constructed_features(individual)
                
                
            if random.random() < self.mutation_probability:
                GeneticUtils \
                        .with_instance(self) \
                        .complex_mutate_original_features(individual)

            GeneticUtils \
                    .with_instance(self) \
                    .fix_individual(individual)
        return new_population

    def elitism(self, population1, population2):
        maximum = max(population1, key=lambda x: x[1])
        minimum_index = min(enumerate(population2),
                            key=lambda x: x[1][1])[0]
        population2[minimum_index] = maximum
        return population2

    def truncation(self, population1, population2):
        return sorted(population1 + population2, reverse=True, key=lambda x: x[1])[:len(population1)]

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

    def copy_individual(self, individual):
        return ([chrms.copy() for chrms in individual[0]], [chrms.copy() for chrms in individual[1]], individual[2].copy())

    def fit(self, X, y):
        self.feature_encoder_ = CustomOrdinalFeatureEncoder()
        self.class_encoder_ = CustomLabelEncoder()

        if isinstance(X, pd.DataFrame):
            self.categories_ = X.columns
        if self.encode_data:
            X = self.feature_encoder_.fit_transform(X)
            y = self.class_encoder_.fit_transform(y)

        classifier_ = NaiveBayes(encode_data=False, n_intervals=self.n_intervals, metric=self.metric)
        self.n_features = X.shape[1]
        if self.encode_data:
            self.unique_values = [values.shape[0] for values in self.feature_encoder_.categories_]
        else:
            self.unique_values = [np.unique(X[:, j]).shape[0] for j in range(X.shape[1])]
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.size = np.ceil(np.sqrt(X.shape[1]))
        best_individual = self.execute_algorithm(X, y)
        self.best_features = best_individual
        self.classifier_ = NaiveBayes(encode_data=False, metric=self.metric)
        self.classifier_.fit(np.concatenate([feature.transform(X) for feature in self.best_features], axis=1), y)
        return self

    def execute_algorithm(self, X, y):
        if self.mixed:
            self.evaluate = self.evaluate_heuristic
        else:
            self.evaluate = self.evaluate_wrapper
        population = self.generate_population()
        population_with_fitness = self.fitness(population, X, y)
        iterator = tqdm(range(self.generations), leave=False) if self.verbose else range(self.generations)
        for generation in iterator:
            if self.mixed and generation > int(self.generations*self.mixed_percentage) and self.evaluate == self.evaluate_heuristic:
                self.evaluate = self.evaluate_wrapper
                # Reevaluate for fair combination
                population_with_fitness = self.fitness([individual_with_fitness[0]
                                                        for individual_with_fitness in population_with_fitness], X, y)
            selected_individuals = self.selection(population_with_fitness)
            crossed_individuals = selected_individuals  # self.crossover(selected_individuals)
            mutated_individuals = self.mutation(crossed_individuals, X=X, y=y)
            new_population = self.fitness(mutated_individuals, X, y)
            population_with_fitness = self.combine(
                population_with_fitness, new_population)

            # Obtaining population's statistics
            if self.verbose:
                best, mean = get_max_mean(population_with_fitness)
                iterator.set_postfix({"Generation": generation,
                                      "hit_count": self.evaluate.hit_count,
                                      "populationLength": len(population_with_fitness),
                                      "best fitness": best,
                                      "mean fitness": mean})

        best_individual = max(population_with_fitness, key=lambda x: x[1])[0]
        return best_individual[0]+best_individual[1]

    def reset_evaluation(self):
        self.evaluate_wrapper = memoize_genetic(self.simple_evaluate)
        self.evaluate_heuristic = memoize_genetic(self.simple_evaluate_heuristic)

    def set_params(self, **params):
        super().set_params(**params)
        if "selection" in params:
            if params["selection"] not in ("rank", "proportionate"):
                raise ValueError("Unknown selection parameter expected one of : " +
                                 str(tuple(["rank", "proportionate"])))
            self.selection = self.select_population_rank if "rank" in params["selection"] else self.select_population
        if "combine" in params:
            if params["combine"] not in ("elitism", "truncate"):
                raise ValueError("Unknown selection parameter expected one of : " + str(tuple(["elitism", "truncate"])))
            self.combine = self.elitism if "elit" in params["combine"] else self.truncation
        if "mutation" in params:
            if params["mutation"] not in ("complex", "simple"):
                raise ValueError("Unknown selection parameter expected one of : " + str(tuple(["complex", "simple"])))
            self.mutation = self.mutate_simple if "simple" == params["mutation"] else self.mutate_complex

    def __init__(self,
                 seed=None,
                 individuals=1,
                 generations=40,
                 mutation_probability=0.2,
                 selection="rank",
                 mutation="simple",
                 combine="elitism",
                 n_intervals=5,
                 metric="accuracy",
                 flexible_logic=True,
                 verbose=False,
                 encode_data=True,
                 mixed=True,
                 mixed_percentage=0.5
                 ):
        self.mixed_percentage = mixed_percentage
        self.mixed = mixed
        self.encode_data = encode_data
        self.flexible_logic = flexible_logic
        self.verbose = verbose
        self.n_intervals = n_intervals
        self.metric = metric
        self.seed = seed
        self.individuals = individuals
        self.generations = generations
        self.mutation_probability = mutation_probability

        self.selection = selection
        self.combine = combine
        self.mutation = mutation

        allowed_selection = ('rank', 'proportionate')
        allowed_combine = ('elitism', 'truncate')
        allowed_mutation = ('complex', 'simple')

        if self.selection not in allowed_selection:
            raise ValueError("Unknown selection type: %s, expected one of %s." % (self.selection, selection))
        if self.combine not in allowed_combine:
            raise ValueError("Unknown combine type: %s, expected one of %s." % (self.combine, combine))
        if self.mutation not in allowed_mutation:
            raise ValueError("Unknown selection type: %s, expected one of %s." % (self.mutation, mutation))

        self.selection = self.select_population_rank if "rank" in selection else self.select_population
        self.combine = self.elitism if "elit" in combine else self.truncation
        self.mutation = self.mutate_simple if "simple" in mutation else self.mutate_complex
        self.reset_evaluation()



class Individual:
    
    def __init__(self):
        self.constructed_features = []
        self.original_features = []
        self.original_features_set = set()
              
    def add_constructed(self, feature):
        self.constructed.append(feature)
        
    def add_original(self, index):
        if index in self.original_features_set:
            return
        self.original_features.append(DummyFeatureConstructor(index))
        self.original_features_set.add(index)
        
    def constructed_features_size(self):
        return len(self.constructed_features)
    
    def has_constructed_features(self):
        return len(self.constructed_features) > 0
    
    def has_original_features(self):
        return len(self.original_features) > 0
        
    def to_tuple(self):
        return (self.constructed_features, self.original_features, self.original_features_set)
        
    def remove_constructed_features(self, indices):
        self.constructed_features = [v for i,v in enumerate(self.constructed_features) if i not in indices]
    
    def remove_original_ifeatures(self, indices):
        self.original_features = [v for i,v in enumerate(self.original_features) if i not in indices]
        self.original_features_set = self.original_features_set - indices
    