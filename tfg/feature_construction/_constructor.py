import numpy as np
import  pandas as pd

from itertools import combinations,product

from tfg.feature_construction import FeatureOperand, FeatureOperator


def get_unique_combinations(X):
#     return np.unique(X,axis=0) -> Slower but might produce less combinations
    return np.array(list(product(*[ pd.unique(X[:,j]) for j in range(X.shape[1])])))

def construct_features(X,operators=('AND','OR')):
    feature_combinations = combinations(np.arange(0,X.shape[1]),2)#combinations_without_repeat(range(X))
    values_combinations = ((index,get_unique_combinations(X[:,index])) for index in feature_combinations)#[(feature_combinations[i],get_uniques(feature_combinations[i],X)) for i in range(feature_combinations.shape[0])]
    constructed_features = product(operators,values_combinations)

    return create_feature_list(constructed_features)
    
def create_feature_list(constructed_features):
    features_list = []
    for constructed_feature in constructed_features:
        operator,distinct_values = constructed_feature
        feature_index, values = distinct_values
        for value in (values[i] for i in range(values.shape[0])):
            feature = create_feature(operator=operator,
                                     operands=list(zip(feature_index,value))
                                     )
            features_list.append(feature)
    return features_list

def create_feature(operator,operands):
    return FeatureOperator(operator=operator,
                           operands=[FeatureOperand(feature_index=index,
                                                    value=value)
                                    for index,value in operands])