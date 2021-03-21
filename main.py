from tfg.executions import scoring_comparison
from tfg.executions import time_comparison
from tfg.utils import get_graphs             
from tfg.executions import ranker_score_comparison


base_path = "../Dataset/UCIREPO/"
data = [
    ["abalone","Rings"],
    ["adult","income"],
    ["anneal","label"],
    ["audiology","label"],
    ["balance-scale","label"],
    ["krkopt","Optimal depth-of-win for White"],
    ["iris","Species"],
    ["horse-colic","surgery"],
    ["glass","Type"],
    ["krkp","label"],
    ["mushroom","class"],
    ["voting","Class Name"],
    ["credit","A16"],
    ["pima","Outcome"],
    ["wine","class"],
    ["wisconsin","diagnosis"],
    ["car-evaluation","safety"],
    ["connect-4","class"],
    ["lenses","ContactLens"],
    ["cmc","Contraceptive"],
    ["cylinder-bands","band type"],
    ["derm","class"],
    ["tictactoe","class"],
    ["spam","class"]
]

graphs_folder = "out/graphs/"
csv_folder = "out/csv/"

#create directories
import os 
for directory in [graphs_folder,csv_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

#Time comparison
# file_name = "time_performace"
# result = time_comparison()
# get_graphs(df = result, 
#            folder =graphs_folder)
# result.to_csv(csv_folder+"time_performace.csv",index=False)

#Scoring comparisons CustomNaiveBayes vs Categorical
# result = scoring_comparison(base_path,datasets=data,test_size=0.3,seed=5,n_iterations=30)
# result.to_csv(csv_folder+"score_comparison_simple_nb.csv",index=False)

#Ranker comparisons
params = [
    {"strategy":"eager","block_size":1,"verbose":0},
    {"strategy":"eager","block_size":2,"verbose":0},
    {"strategy":"eager","block_size":5,"verbose":0},
    {"strategy":"eager","block_size":10,"verbose":0},
    {"strategy":"skip","block_size":1,"max_iterations":10,"verbose":0},
    {"strategy":"skip","block_size":2,"max_iterations":10,"verbose":0},
    {"strategy":"skip","block_size":5,"max_iterations":10,"verbose":0},
    {"strategy":"skip","block_size":10,"max_iterations":10,"verbose":0},
    {"strategy":"skip","block_size":1,"max_features":20,"verbose":0},
    {"strategy":"skip","block_size":2,"max_features":20,"verbose":0},
    {"strategy":"skip","block_size":5,"max_features":20,"verbose":0},
    {"strategy":"skip","block_size":10,"max_features":20,"verbose":0},
]
result = ranker_score_comparison(base_path=base_path,
                                 datasets=data,
                                 test_size=0.3,
                                 seed=5,
                                 n_iterations=30,
                                 params = params)
result.to_csv(csv_folder+"ranker_score_comparison.csv",index=False)