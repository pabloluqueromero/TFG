
from tfg.feature_construction import DummyFeatureConstructor, create_feature

import GeneticProgrammingFlexibleLogic


class GeneticUtils:
    
    @staticmethod
    def with_instance(algorithm: GeneticProgrammingFlexibleLogic):
        return GeneticUtils(algorithm)
    
    def __init__(self, algorithm: GeneticProgrammingFlexibleLogic):
        self.random = algorithm.random
        self.n_features = algorithm.n_features
        self.flexible_logic = algorithm.flexible_logic
        self.unique_values = algorithm.unique_values
        self.size = algorithm.size
        
    def get_inidividual_size(self):
        if self.algorithm.flexible_logic:
            return  range(self.algorithm.self.random.randint(1, self.algorithm.size)) 
        else:
            return range(self.algorithm.size)
    
    def generate_feature(self):
        operand1_feature = self.self.random.randint(0, self.n_features-1)
        operand2_feature = self.self.random.randint(0, self.n_features-1)
        if operand1_feature == operand2_feature:
            op = 'OR'
            operand1_value = self.self.random.randint(0, self.unique_values[operand1_feature]-1)
            operand2_value = self.self.random.randint(0, self.unique_values[operand1_feature]-1)
        else:
            op = self.self.random.choice(('OR', 'XOR', 'AND'))
            operand1_value = self.self.random.randint(0, self.unique_values[operand1_feature]-1)
            operand2_value = self.self.random.randint(0, self.unique_values[operand2_feature]-1)
        operands = []
        operands.append((operand1_feature, operand1_value))
        operands.append((operand2_feature, operand2_value))
        return create_feature(operator=op, operands=operands)
    
    def _alter_feature(self, feature):
        feature.op = self.self.random.choice(('OR', 'XOR', 'AND'))
        for operand in feature.operands:
            operand.feature_index = self.self.random.randint(0, self.n_features-1)
            operand.value = self.self.random.randint(0, self.unique_values[operand.feature_index]-1)
    
    def _create_random_feature(self):
        op = self.self.random.choice(('OR', 'XOR', 'AND'))
        operands = []
        for _ in range(2):
            feature_index = self.self.random.randint(0, self.n_features-1)
            value = self.self.random.randint(0, self.unique_values[feature_index]-1)
            operands.append((feature_index, value))
        return create_feature(operator=op, operands=operands)
        
    def complex_mutate_constructed_features(self, individual):
        if self.flexible_logic and not individual.has_constructed_features():
            #We can mutate only by adding a feature
            feature = self.create_random_feature()
            individual.add_constructed_feature(feature)
            return individual

        features_to_change = self.self.random.sample(
                list(range(individual.constructed_features_size())), self.self.random.randint(1, individual.constructed_features_size()))

        indices_to_remove = set()
        for feature_index in features_to_change:
            if not self.flexible_logic:
                feature = individual.constructed_features[feature_index]
                self._alter_feature(feature)
            else:
                p = self.self.random.random()
                if p < 0.33:
                    #Modify a feature (only operands)
                    self._alter_feature(feature)
                elif p < 0.66:
                    #Add a feature
                    feature = self.create_random_feature()
                    individual.add_constructed_feature(feature)
                else:
                    #Delete a feature
                    indices_to_remove.add(feature_index)
        individual.remove_constructed_features(indices_to_remove)
        return individual
    
    def complex_mutate_original_features(self, individual):
        a = self.random.random()
        og_features = individual.original_features
        included_features = individual.original_feature_set
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
        return individual
    
    def fix_individual(self, individual):
        #If an individual is comletely empty we add a single feature to it
        if not individual.has_constructed_features() and not individual.has_original_features():
            feature_index= self.random.choice(range(0, self.n_features))
            individual.add_original(feature_index)
            
    def simple_mutate_constructed_features(self, individual):
        chromosomes_index = []
        if self.flexible_logic and not individual.has_constructed_features():
            #We can mutate only by adding a feature
            feature = self.create_random_feature()
            individual.add_constructed_feature(feature)
            return individual

        chromosomes_index = self.random.sample(
                list(range(len(individual[1]))), self.random.randint(1, len(individual[1])))
        
        indices_to_remove = set()
        for i in range(len(chromosomes_index)):
            index = chromosomes_index[i]
            feature = individual[1][index]
            if not self.flexible_logic:
                feature.op = self.random.choice(('OR', 'XOR', 'AND'))
                for operand in feature.operands:
                    operand.feature_index = self.random.randint(0, self.n_features-1)
                    operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
            else:
                a = self.random.random()
                if a < 0.33:
                    b = self.random.random()
                    if b < 0.2:
                        # Change operatior
                        feature.op = self.random.choice(('OR', 'XOR', 'AND'))
                    elif b < 0.4:
                        # Change full operand
                        operand = feature.operands[0]
                        operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
                    elif b < 0.6:
                        # Change full operand
                        operand = feature.operands[1]
                        operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
                    elif b < 0.8:
                        # Change value
                        operand = feature.operands[0]
                        operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)
                    else:
                        # Change value
                        operand = feature.operands[1]
                        operand.value = self.random.randint(0, self.unique_values[operand.feature_index]-1)

                elif a < 0.66:
                    # Add feature
                    feature = self.create_random_feature()
                    individual.add_constructed_feature(feature)

                else:
                    # Remove feature
                    indices_to_remove.add(index)
        individual.remove_original_features(indices_to_remove)