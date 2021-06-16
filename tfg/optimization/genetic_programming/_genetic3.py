
import math
import random

import numpy as np
import pandas as pd

from itertools import chain
from time import time
from tqdm.autonotebook import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor, create_feature
from tfg.naive_bayes import NaiveBayes
from tfg.optimization import OptimizationMixin
from tfg.optimization.genetic_programming import GeneticProgrammingFlexibleLogic
from tfg.utils import compute_sufs, compute_sufs_non_incremental
from tfg.utils import memoize_genetic, symmetrical_uncertainty, transform_features
from tfg.utils import get_max_mean


class GeneticProgrammingRankMutation(GeneticProgrammingFlexibleLogic):
    """ GeneticProgrammingRankMutation is similar adds a guided-mutation by creating a ranking with features based on
        Symmetrical Uncertainty.
        Refer to GeneticProgrammingFlexibleLogic for parameter specification.
        The flexible logic is always set to True
    """

    def single_feature(self, feature, X, y):
        data = feature.transform(X)
        return symmetrical_uncertainty(data, y)

    def smart_random_sample(self, features, size, X, y):
        '''
        Creates a descending ranking of the features based on the SU(feature,y)
        The probability of being chosen is equals to:
            position_rank(feature)/feature.shape[0]
        '''
        array = np.array([self.single_feature_evaluation(feature, X, y) for feature in features])
        index_array = np.argsort(array)[::-1]  # Descending order
        total = array.shape[0]*(array.shape[0]+1)/2
        probability = np.empty_like(array)
        probability[index_array] = (np.arange(1, array.shape[0]+1)/total)
        return np.random.choice(np.arange(0, len(features)), size=size, replace=False, p=probability)

    def mutation_rank(self, population, X, y):
        new_population = []
        for individual in population:
            if random.random() < self.mutation_probability:
                chromosomes_index = []
                if len(individual[1]) == 0:
                    op = random.choice(('OR', 'XOR', 'AND'))
                    operands = []
                    for _ in range(2):
                        feature_index = random.randint(0, self.n_features-1)
                        value = random.randint(0, self.unique_values[feature_index]-1)
                        operands.append((feature_index, value))
                    individual[1].append(create_feature(operator=op, operands=operands))
                    new_population.append(individual)
                    continue

                else:
                    size = random.randint(1, len(individual[1]))
                    chromosomes_index = self.smart_random_sample(individual[1], size, X, y)

                for i in range(len(chromosomes_index)):
                    index = chromosomes_index[i]
                    feature = individual[1][index]
                    a = random.random()
                    if a < 0.33:
                        b = random.random()
                        if b < 0.2:
                            # Change operatior
                            feature.op = random.choice(('OR', 'XOR', 'AND'))
                        elif b < 0.4:
                            # Change full operand
                            operand = feature.operands[0]
                            operand.value = random.randint(0, self.unique_values[operand.feature_index]-1)
                        elif b < 0.6:
                            # Change full operand
                            operand = feature.operands[1]
                            operand.value = random.randint(0, self.unique_values[operand.feature_index]-1)
                        elif b < 0.8:
                            # Change value
                            operand = feature.operands[0]
                            operand.value = random.randint(0, self.unique_values[operand.feature_index]-1)
                        else:
                            # Change value
                            operand = feature.operands[1]
                            operand.value = random.randint(0, self.unique_values[operand.feature_index]-1)

                    elif a < 0.66:
                        # Add feature
                        op = random.choice(('OR', 'XOR', 'AND'))
                        operands = []
                        for _ in range(2):
                            feature_index = random.randint(0, self.n_features-1)
                            value = random.randint(0, self.unique_values[feature_index]-1)
                            operands.append((feature_index, value))
                        individual[1].append(create_feature(operator=op, operands=operands))

                    else:
                        # Remove feature
                        del individual[1][index]
                        chromosomes_index = [j-1 if j > index else j for j in chromosomes_index]

            if random.random() < self.mutation_probability:
                a = random.random()
                og_features = individual[0]
                included_features = individual[2]
                if (a < 0.33 and len(og_features) < self.n_features) or len(og_features) == 0:
                    selected = random.choice(tuple(set(list(range(0, self.n_features))) - included_features))
                    included_features.add(selected)
                    og_features.append(DummyFeatureConstructor(selected))
                elif a < 0.66 and len(og_features) < self.n_features and len(og_features) > 0:
                    selected = random.choice(tuple(set(list(range(0, self.n_features))) - included_features))
                    index = random.randint(0, len(og_features)-1)
                    feature = og_features[index].feature_index
                    og_features[index] = DummyFeatureConstructor(selected)
                    included_features.remove(feature)
                    included_features.add(selected)
                else:
                    index = random.randint(0, len(og_features)-1)
                    feature = og_features[index].feature_index
                    del og_features[index]
                    included_features.remove(feature)

            if len(individual[0]) == 0 and len(individual[1]) == 0:
                og_features = individual[0]
                included_features = individual[2]
                selected = random.choice(tuple(set(list(range(0, self.n_features))) - included_features))
                included_features.add(selected)
                og_features.append(DummyFeatureConstructor(selected))
            new_population.append(individual)
        return new_population

    def reset_evaluation(self):
        self.evaluate_wrapper = memoize_genetic(self.simple_evaluate)
        self.evaluate_heuristic = memoize_genetic(self.simple_evaluate_heuristic)
        self.single_feature_evaluation = memoize_single(self.single_feature)

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
        self.mutation = self.mutation_rank
        self.flexible_logic = True

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
                 verbose=False,
                 encode_data=True,
                 mixed=True,
                 mixed_percentage=0.5
                 ):
        self.mixed_percentage = mixed_percentage
        self.mixed = mixed
        self.encode_data = encode_data
        self.flexible_logic = True
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

        if self.selection not in allowed_selection:
            raise ValueError("Unknown selection type: %s, expected one of %s." % (self.selection, selection))
        if self.combine not in allowed_combine:
            raise ValueError("Unknown combine type: %s, expected one of %s." % (self.combine, combine))

        self.selection = self.select_population_rank if "rank" in selection else self.select_population
        self.combine = self.elitism if "elit" in combine else self.truncation
        self.mutation = self.mutation_rank
        self.flexible_logic = True
        self.reset_Evaluation

def memoize_single(f):
    cache = dict()
    def g(individual, X, y):
        hash_individual = hash(individual)
        if hash_individual not in cache:
            cache[hash_individual] = f(individual, X, y)
        return cache[hash_individual]
    return g
