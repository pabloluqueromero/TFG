from tfg.feature_construction._constructor import construct_features
from tfg.utils import get_X_y_from_database, symmetrical_uncertainty
from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
import os
import numpy as np
import math as m
from sklearn.model_selection import RepeatedStratifiedKFold
from itertools import combinations
import heapq
from tfg.utils._utils import symmetrical_uncertainty_two_variables


base_path = "./UCIREPO/"
databases = [
    ["yeast","nuc"],
    ["abalone","Rings"],
    ["mammographicmasses","Label"],
    ["breast-cancer","Class"],
    ["anneal","label"],
    ["audiology","label"],
    ["balance-scale","label"],
    ["iris","Species"],
    ["student","Walc"],
    ["electricgrid","stabf"],
    ["horse-colic","surgery"],
    ["glass","Type"],
    ["krkp","label"],
    ["lenses","ContactLens"],
    ["mushroom","class"],
    ["voting","Class Name"],
    ["credit","A16"],
    ["pima","Outcome"],
    ["wine","class"],
    ["wisconsin","diagnosis"],
    ["car-evaluation","safety"],
    ["cmc","Contraceptive"],
    ["cylinder-bands","band type"],
    ["derm","class"],
    ["tictactoe","class"],
    ["spam","class"]
]


avg_features = []
avg_position = []

for database in databases:
    name, label = database
    if os.path.exists(base_path+name):
        test = f"{name}.test.csv"
        data = f"{name}.data.csv"
        X, y = get_X_y_from_database(base_path, name, data, test, label)
    c = CustomOrdinalFeatureEncoder(n_intervals = 5)
    l = CustomLabelEncoder()
    X = c.fit_transform(X)
    y = l.fit_transform(y)


    original_features_in_top = set()
    features = heapq.nlargest(3, ((symmetrical_uncertainty_two_variables(X[:, i], X[:, j], y), (i, j)) for i, j in combinations(range(X.shape[1]), 2)),key=lambda x: x[0])
    for score, feature in features:
        original_features_in_top.add(feature[0])
        original_features_in_top.add(feature[1])
    all_feature_constructors = construct_features(X,operators=('OR','XOR','AND'))
    symmetrical_uncertainty_rank = []
    for feature_constructor in all_feature_constructors:
            feature = feature_constructor.transform(X)
            su = symmetrical_uncertainty(f1=feature,f2=y)
            symmetrical_uncertainty_rank.append(su)
    rank = np.argsort(symmetrical_uncertainty_rank)[::-1] #Descending order

    # Get top 5
    # top_features = rank[:5]
    # original_features_in_top = set()
    # for feature in map(lambda index: all_feature_constructors[index],top_features):
    #     for operand in feature.operands:
    #         original_features_in_top.add(operand.feature_index)

    position = {feature : [] for feature in original_features_in_top}
    for i,feature in enumerate(map(lambda index: all_feature_constructors[index],rank)):
        for operand in filter(lambda x: x.feature_index in original_features_in_top,feature.operands):
            position[operand.feature_index].append(i)

    print("Database:", database)
    print(f"\t\tRank length: {len(rank)}")
    for feature_index,list_index in position.items():
        print(f"\t\t FEATURE {feature_index}: AVG INDEX: {np.mean(list_index)} STD INDEX: {np.std(list_index)}")
    avg_features.append(np.mean(list_index) / len(rank))
    avg_position.append(len(original_features_in_top) / min(10,X.shape[1]))
    
print("Average TOP: ",np.mean(avg_features))
print("Average POS: ",np.mean(avg_position))