import numpy as np

from itertools import combinations,product

from tfg.feature_construction import FeatureOperand, FeatureOperator
from tfg.utils import combinations_without_repeat


def get_uniques(index,array):
    return np.unique(array[:,index],axis=0)

def construct_features(X,operators=('AND','OR')):
    feature_combinations = np.array(list(combinations(np.arange(0,X.shape[1]),2)))#combinations_without_repeat(range(X))
    values_combinations =[(feature_combinations[i],get_uniques(feature_combinations[i],X)) for i in range(feature_combinations.shape[0])]
    constructed_features = list(product(operators,values_combinations))

    features_list = []
    for constructed_feature in constructed_features:
        operator,distinct_values = constructed_feature
        feature_index, values = distinct_values
        for value in (values[i] for i in range(values.shape[0])):
            feature = create_feature(operator=operator,
                                     operands=[
                                        zip(feature_index,value)
                                     ])
    print(feature)
            

def create_feature(operator,operands):
    return FeatureOperator(operator=operator,
                           operands=[FeatureOperand(feature_index=index,
                                                    value=value)
                                    for index,value in operands])