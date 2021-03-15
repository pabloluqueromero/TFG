import numpy as np
import  pandas as pd

from itertools import combinations,product

from tfg.feature_construction import FeatureOperand, FeatureOperator

def get_unique_combinations(X):
    '''Gets all the possible pairs of values vi and vj such that vi belong to Xi and vj belongs to Xj for every pair of features '''
    return np.unique(X,axis=0) #-> Slower but might produce less combinations (only combinations that appear in the database)
    # return list(product(*[ pd.unique(X[:,j]) for j in range(X.shape[1])])) # All possible combinations even though some might not appear

def construct_features(X,operators=('AND','OR','XOR')):
    '''For each combination returned by get_unique_combinations it generates 3 features with the specified operators)'''
    feature_combinations = combinations(np.arange(0,X.shape[1]),2)#combinations_without_repeat(range(X))
    values_combinations = ((index,get_unique_combinations(X[:,index])) for index in feature_combinations)#[(feature_combinations[i],get_uniques(feature_combinations[i],X)) for i in range(feature_combinations.shape[0])]
    constructed_features = product(operators,values_combinations)

    return create_feature_list(constructed_features)
    
def create_feature_list(constructed_features):
    '''Auxiliary method that returns the constructors that are objects that inherit from Feature
    The expected input should have the shape:
        - [ (operator,((feature_index1,feature_index2), (value1,value2))),
            ...
          ]
    '''
    features_list = []
    for constructed_feature in constructed_features:
        operator,distinct_values = constructed_feature
        feature_index, values = distinct_values
        for value in values:
            feature = create_feature(operator=operator,
                                     operands=list(zip(feature_index,value))
                                     )
            features_list.append(feature)
    return features_list

def create_feature(operator,operands):
    ''''Creates the feature'''
    return FeatureOperator(operator=operator,
                           operands=[FeatureOperand(feature_index=index,
                                                    value=value)
                                    for index,value in operands])