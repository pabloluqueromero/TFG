import numpy as np

from collections import OrderedDict
from numba import njit
from sklearn.base import TransformerMixin


class Feature(TransformerMixin):
    def get_dict_translation(self,encoder=None,categories=None):
        pass

class FeatureOperator(Feature):
    def __init__(self,operator,operands):
        self.operator = operator
        self.operands = operands
        allowed_operators = ("AND","OR","XOR")
        if self.operator not in allowed_operators:
            raise ValueError("Unknown operator type: %s, expected one of %s." % (self.operator, allowed_operators))
        self._get_operator(self.operator)

    def fit(self,X):
        return X

    def transform(self,X):
        return self.operator_callable(*[operand.transform(X) for operand in self.operands]).reshape(-1,1)

    def _get_operator(self,operator):
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
            s += child.to_str(depth+1)
        return s
    
    def get_dict_translation(self,encoder=None,categories=None):
        od = OrderedDict()
        od["operator"] = self.operator
        for i,operand in enumerate(self.operands):
            od[f"operand_{i}"] = operand.get_dict_translation(encoder,categories)
        return od



class DummyFeatureConstructor(Feature):
    def __init__(self,feature_index):
        self.feature_index = feature_index
    def fit(self,X):
        return X

    def transform(self,X):
        return X[:,self.feature_index].reshape(-1,1)

    def to_str(self, depth=0):
        return  '\t' * depth + f"FeatureDummy:  index-> {self.feature_index}\n"

    def get_dict_translation(self,encoder=None,categories=None):
        return {"feature":categories[int(self.feature_index)] if categories is not None else int(self.feature_index)}

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

    def get_dict_translation(self,encoder=None,categories=None):
        value = self.value
        if encoder:
            value = encoder.inverse_transform_element(self.feature_index,self.value)
            if isinstance(value,np.ndarray):
                value = list(value)
        od = OrderedDict()
        od["feature"] = categories[int(self.feature_index)] if categories is not None else int(self.feature_index)
        od["value"] = value
        return od