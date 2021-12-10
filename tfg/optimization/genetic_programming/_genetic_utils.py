
from tfg.feature_construction import DummyFeatureConstructor, create_feature
from tfg.utils import symmetrical_uncertainty

import numpy as np

class GeneticUtils:
    
    @staticmethod
    def with_instance(algorithm):
        return GeneticUtils(algorithm)
    
    def __init__(self, algorithm):
        self.random = algorithm.random
        self.n_features = algorithm.n_features
        self.flexible_logic = algorithm.flexible_logic
        self.unique_values = algorithm.unique_values
        self.size = algorithm.size
        
    def get_inidividual_size(self):
        if self.flexible_logic:
            return  range(self.random.randint(1, self.size)) 
        else:
            return range(self.size)
    
    def generate_feature(self):
        operand1_feature = self.random.randint(0, self.n_features-1)
        operand2_feature = self.random.randint(0, self.n_features-1)
        if operand1_feature == operand2_feature:
            op = 'OR'
            operand1_value = self.random.randint(0, self.unique_values[operand1_feature]-1)
            operand2_value = self.random.randint(0, self.unique_values[operand1_feature]-1)
        else:
            op = self.random.choice(('OR', 'XOR', 'AND'))
            operand1_value = self.random.randint(0, self.unique_values[operand1_feature]-1)
            operand2_value = self.random.randint(0, self.unique_values[operand2_feature]-1)
        operands = []
        operands.append((operand1_feature, operand1_value))
        operands.append((operand2_feature, operand2_value))
        return create_feature(operator=op, operands=operands)
    
    def _alter_feature(self, feature):
        feature.op = self.random.choice(('OR', 'XOR', 'AND'))
        for operand in feature.operands:
            operand.feature_index = self.random.randint(0, self.n_features-1)
            operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
    
    def _create_random_feature(self):
        op = self.random.choice(('OR', 'XOR', 'AND'))
        operands = []
        for _ in range(2):
            feature_index = self.random.randint(0, self.n_features-1)
            value = self.random.randint(0, self.unique_values[feature_index]-1)
            operands.append((feature_index, value))
        return create_feature(operator=op, operands=operands)
        
    def simple_mutate_constructed_features(self, individual, smart = False):
        if self.flexible_logic and not individual.has_constructed_features():
            #We can mutate only by adding a feature
            feature = self._create_random_feature()
            individual.add_constructed_feature(feature)
            return

        features_to_change = self.random.sample(
                list(range(individual.constructed_features_size())), self.random.randint(1, individual.constructed_features_size()))

        indices_to_remove = set()
        for feature_index in features_to_change:
            if not self.flexible_logic:
                    #Modify a feature (only operands)
                feature = individual.constructed_features[feature_index]
                self._alter_feature(feature)
            else:
                p = self.random.random()
                if p < 0.33:
                    #Modify a feature (only operands)
                    feature = individual.constructed_features[feature_index]
                    self._alter_feature(feature)
                elif p < 0.66:
                    #Add a feature
                    feature = self._create_random_feature()
                    individual.add_constructed_feature(feature)
                else:
                    #Delete a feature
                    indices_to_remove.add(feature_index)
        individual.remove_constructed_features(indices_to_remove)
    
    def mutate_original_features(self, individual):
        a = self.random.random()
        og_features = individual.original_features
        included_features = individual.original_features_set
        if (a < 0.33 and len(og_features) < self.n_features) or len(og_features) == 0:
            selected = self.random.choice(tuple(set(list(range(0, self.n_features))) - included_features))
            included_features.add(selected)
            og_features.append(DummyFeatureConstructor(selected))
        elif a < 0.66 and len(og_features) < self.n_features and len(og_features) > 0:
            selected = self.random.choice(tuple(set(list(range(0, self.n_features))) - included_features))
            index = self.random.randint(0, len(og_features)-1)
            feature = og_features[index].feature_index
            og_features[index] = DummyFeatureConstructor(selected)
            included_features.remove(feature)
            included_features.add(selected)
        else:
            index = self.random.randint(0, len(og_features)-1)
            feature = og_features[index].feature_index
            del og_features[index]
            included_features.remove(feature)
    
    def fix_individual(self, individual):
        #If an individual is comletely empty we add a single feature to it
        if not individual.has_constructed_features() and not individual.has_original_features():
            feature_index= self.random.choice(range(0, self.n_features))
            individual.add_original(feature_index)
            
    def complex_mutate_constructed_features(self, individual):
        if self.flexible_logic and not individual.has_constructed_features():
            #We can mutate only by adding a feature
            feature = self._create_random_feature()
            individual.add_constructed_feature(feature)
            return
        
        features_to_change = self.random.sample(
                list(range(individual.constructed_features_size())), self.random.randint(1, individual.constructed_features_size()))

        indices_to_remove = set()
        for feature_index in features_to_change:
            if not self.flexible_logic:
                feature.op = self.random.choice(('OR', 'XOR', 'AND'))
                for operand in feature.operands:
                    operand.feature_index = self.random.randint(0, self.n_features-1)
                    operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
            else:
                p = self.random.random()
                if p < 0.33:
                    #Apply a fine grained modification
                    self._fine_grained_modification(individual.constructed_features[feature_index])
                elif p < 0.66:
                    # Add feature
                    feature = self._create_random_feature()
                    individual.add_constructed_feature(feature)

                else:
                    # Remove feature
                    indices_to_remove.add(feature_index)
        individual.remove_original_features(indices_to_remove)
        
    def _fine_grained_modification(self, feature):
        b = self.random.random()
        if b < 0.2:
            # Change operatior
            feature.op = self.random.choice(('OR', 'XOR', 'AND'))
        elif b < 0.4:
            # Change full left operand
            operand = feature.operands[0]
            operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
        elif b < 0.6:
            # Change full right operand
            operand = feature.operands[1]
            operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
        elif b < 0.8:
            # Change value right operand
            operand = feature.operands[0]
            operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
        else:
            # Change value left operand
            operand = feature.operands[1]
            operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)

    def rank_mutate_constructed_features(self, individual, X, y):
        if self.flexible_logic and not individual.has_constructed_features():
            #We can mutate only by adding a feature
            feature = self._create_random_feature()
            individual.add_constructed_feature(feature)
            return
        
        sample_size = self.random.randint(1, individual.constructed_features_size())
        features_to_change = self.smart_random_sample(individual.constructed_features, sample_size, X, y)

        indices_to_remove = set()
        for feature_index in features_to_change:
            if not self.flexible_logic:
                feature.op = self.random.choice(('OR', 'XOR', 'AND'))
                for operand in feature.operands:
                    operand.feature_index = self.random.randint(0, self.n_features-1)
                    operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
            else:
                p = self.random.random()
                if p < 0.33:
                    #Apply a fine grained modification
                    self._fine_grained_modification(individual.constructed_features[feature_index])
                elif p < 0.66:
                    # Add feature
                    feature = self._create_random_feature()
                    individual.add_constructed_feature(feature)

                else:
                    # Remove feature
                    indices_to_remove.add(feature_index)
        individual.remove_original_features(indices_to_remove)
        
    def single_feature_evaluation(self, feature, X, y):
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
