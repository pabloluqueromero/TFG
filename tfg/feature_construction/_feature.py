import numpy as np

from numba import njit
from sklearn.base import TransformerMixin


class Feature(TransformerMixin):
    pass

class FeatureOperator(Feature):
    def __init__(self,operator,operands):
        self.operator = operator
        self.operands = operands
        allowed_operators = ("AND","OR","XOR")
        if self.operator not in allowed_operators:
            raise ValueError("Unknown operator type: %s, expected one of %s." % (self.strategy, allowed_operators))
        self.get_operator(self.operator)

    def fit(self,X):
        return X

    def transform(self,X):
        return self.operator_callable(*[operand.transform(X) for operand in self.operands]).reshape(-1,1)

    def get_operator(self,operator):
        if operator == "AND":
            self.operator_callable = np.logical_and
        elif operator == "OR":
            self.operator_callable = np.logical_or
        elif operator == "XOR":
            self.operator_callable = np.logical_xor
        else:
            raise Exception("No matching operator function found")
        
    def to_str(self, depth=0):
        s =  '\t' * depth + f"FeatureOperator: {self.operator}\n"
        for child in self.operands:
            s += child.print(depth+1)
        return s




@njit
def _transform_leaf_node(X,feature_index,value):
    return (X[:,feature_index]==value).reshape(-1,1)

class FeatureOperand(Feature):
    def __init__(self,feature_index,value):
        self.feature_index = feature_index
        self.value = value

    def fit(self,X):
        return X

    def transform(self,X):
        return _transform_leaf_node(X,self.feature_index,self.value)

    def to_str(self, depth=0):
        return  '\t' * depth + f"FeatureOperand:  index-> {self.feature_index}, value -> {self.value}\n"