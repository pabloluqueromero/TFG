from tfg.executions import *
import subprocess
import argparse
from time import sleep
import numpy as np
import itertools
from tfg.utils._utils import get_X_y_from_database

'''
Global Variables
'''
seed = 200
n_repeats = 5
n_splits = 3
n_intervals = 5
base_path = "./UCIREPO/"
datasets = [
    ["lenses", "ContactLens"],
    ["abalone", "Rings"],
    ["anneal", "label"],
    ["audiology", "label"],
    ["balance-scale", "label"],
    ["breast-cancer", "Class"],
    ["car-evaluation", "safety"],
    ["cmc", "Contraceptive"],
    ["credit", "A16"],
    ["cylinder-bands", "band type"],
    # ["hill_valley", "class"],
    ["derm", "class"],
    ["electricgrid", "stabf"],
    ["glass", "Type"],
    ["horse-colic", "surgery"],
    ["iris", "Species"],
    ["krkp", "label"],
    ["mammographicmasses", "Label"],
    ["mushroom", "class"],
    ["pima", "Outcome"],
    ["student", "Walc"],
    ["voting", "Class Name"],
    ["wine", "class"],
    ["wisconsin", "diagnosis"],
    ["yeast", "nuc"],
    ["tictactoe", "class"],
    ["spam", "class"]
]



graphs_folder = "out/graphs/"
csv_folder = "out/csv/"


#######################Initiate the parser##########################
####################################################################
####################################################################
####################################################################
####################################################################
parser = argparse.ArgumentParser()

parser.add_argument("--email", required=True, help="email")
parser.add_argument("--password", required=True, help="password")
parser.add_argument("--algorithm", required=True,
                    help="algorithm (ranker, genetic, aco)")
parser.add_argument("--method", default=1, help="for ranker and genetic 1-3")
parser.add_argument("--n_computers", required=True, help="")
parser.add_argument("--computer", required=True, help="computer/n_computers")
parser.add_argument("--metric", required=True, default="accuracy",help="scorer")
parser.add_argument("--filename", default="",help="suffix for the")
parser.add_argument("--no_email", action="store_true", help="dont_send_email")
parser.add_argument("--numerical", action="store_true", help="dont_send_email")

args = parser.parse_args()
email = args.email
password = args.password
algorithm = args.algorithm
method = int(args.method)
n_computers = int(args.n_computers)
computer = int(args.computer)-1
metric = args.metric
send_email_cond = not args.no_email
filter_numeric = args.numerical
filename_suffix = args.filename


if filter_numeric:
    filtered_data = []
    for name,label in datasets:
        test = f"{name}.test.csv"
        data = f"{name}.data.csv"
        X, _ = get_X_y_from_database(base_path, name, data, test, label)
        if X.select_dtypes("float").shape[1]>0:
            # print(X.select_dtypes("float").shape[1])
            filtered_data.append((name,label))
    datasets = filtered_data

    
# l = ['abalone', 'anneal', 'audiology', 'balance-scale', 'breast-cancer', 'car-evaluation', 'cmc']
# datasets = list(filter(lambda x: x[0] in l, datasets))

if computer >= n_computers:
    print("ERROR: computer >= n_computers")
    exit(1)


email_data = {
    "FROM": email,
    "TO": email,
    "PASSWORD": password,
}

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

def execute_commands(commands):
    for command in commands:
        print("\nExecuting: - {"+command+"}")
        subprocess.call(command)
        sleep(3)


def setup():
    print("\n------------------------Setting up environment--------------------------------\n")
    commands = [
        # 'rm -rf main.py TFG tfg2 TFGTEMP UCIREPO out Readme.md',
        # "git clone https://github.com/pabloluqueromero/TFG.git",
        # "rm -rf TFG/.git",
        # "mv TFG/tfg TFG/tfg2",
        # "mv TFG/UCIREPO . ",
        # "mv TFG/tfg2 . ",
        # "rm -rf TFG",
        # "mv tfg2 tfg",
        # 'pip uninstall --yes scikit-learn',
        # 'pip install scikit-learn==0.24.1',
    ]
    execute_commands(commands)


def verify_email():
    from tfg.utils import EmailSendCSV

    print("Verifying email . . . . . . . .", end=" ")
    print("OK!")
    EmailSendCSV(email, email, password).send_test()


def finish():
    print("\n------------------------Removing environment--------------------------------\n")
    commands = [
        # 'rm -rf *',
        # 'history -c',
        # 'shutdown now'
    ]

    execute_commands(commands)


def execute():
    datas = np.array_split(datasets, n_computers)
    data = datas[computer-1]

    for d in data:
        print(d)
    if algorithm == "ranker":
        if method == 1:
            execute_ranker_1(data)
        elif method == 2:
            execute_ranker_2(data)
        elif method == 3:
            execute_ranker_3(data)
        elif method == 4:
            execute_ranker_4(data)
        else:
            execute_ranker_5(data)

    elif algorithm == "genetic":
        if method == 1:
            execute_genetic_1(data)
        elif method == 2:
            execute_genetic_2(data)
        elif method == 3:
            execute_genetic_3(data)
    elif algorithm == "aco":
        if method == 1:
            execute_aco_1(data)
        elif method == 2:
            execute_aco_2(data)
        elif method == 3:
            execute_aco_3(data)


def execute_genetic_1(data):
    print("GENETIC 1")
    params = [
        {
            "size": 11,
            "seed": seed,
            "individuals": 20,
            "generations": 20,
            "mutation_probability": 0.3,
            "selection": "simple",
            "mutation": "simple",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": metric,
            "verbose": True,
            "flexible_logic": True,
            "encode": False
        },
        {
            "mutation": "complex",
            "size": 10,
            "seed": seed,
            "individuals": 20,
            "generations": 20,
            "mutation_probability": 0.4,
            "selection": "simple",
            "combine": "truncation",
            "n_intervals": 5,
            "metric": metric,
            "encode": False,
            "verbose": True,
            "flexible_logic": True,
        }, {
            "size": 6,
            "seed": seed,
            "individuals": 20,
            "generations": 20,
            "mutation_probability": 0.2,
            "selection": "rank",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": metric,
            "verbose": True,
            "flexible_logic": False,
        }, {
            "size": 3,
            "seed": seed,
            "individuals": 20,
            "generations": 20,
            "mutation_probability": 0.2,
            "selection": "rank",
            "combine": "elitism",
            "n_intervals": 5,
            "encode": False,
            "metric": metric,
            "flexible_logic": True,
            "verbose": True
        }, {
            "size": 30,
            "seed": seed,
            "individuals": 50,
            "generations": 20,
            "mutation_probability": 0.2,
            "selection": "rank",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": metric,
            "flexible_logic": True,
            "verbose": True,
            "encode": False
        },
    ]

    for data_i in data[::1]:
        try:
            result = genetic_score_comparison(base_path=base_path,
                                              datasets=[data_i],
                                              n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              seed=seed,
                                              params=params,
                                              n_intervals=n_intervals,
                                              metric=metric,
                                              send_email=send_email_cond,
                                              version=2,
                                              email_data={**email_data,
                                                          **{
                                                              "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                              "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}                                                          })
            result.to_csv(
                f"final_result/genetic_1/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")


def execute_genetic_2(data):
    print("GENETIC 2")

    grid = {
        "mutation_probability": [0.05,0.1,0.2],
        "selection": ["rank","proportionate"],
        "combine": ["elitism","truncate"],
        "mixed": [True],
        "backwards":[False]
    }

    def_params = {
            "size":np.nan,
            "seed": seed,
            "individuals": 30,
            "generations": 20,
            "mutation_probability": 0.01,
            "selection": "proportionate",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": metric,
            "verbose": False,
            "mixed": False,
            "encode": False,
            "mixed_percentage": 0.8

    }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    print("Conf Size: ",len(params))

    for data_i in data[::1]:
        try:
            result = genetic_score_comparison(base_path=base_path,
                                              datasets=[data_i],
                                              n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              seed=seed,
                                              metric = metric,
                                              params=params,
                                              n_intervals=n_intervals,
                                              send_email=send_email_cond,
                                              version=2,
                                              email_data={**email_data,
                                                          **{
                                                              "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                              "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}                                                          })
            result.to_csv(
                f"final_result/genetic_2/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")

def execute_genetic_3(data):
    print("GENETIC 3")
    grid = {
        "mutation_probability": [0.01,0.05,0.1,0.2],
        "selection": ["rank","proportionate"],
        "combine": ["truncate"],
        "mixed": [True,False],
        "backwards":[False]
    }
    def_params = {
            "size":np.nan,
            "seed": seed,
            "individuals": 50,
            "generations": 30,
            "mutation_probability": 0.01,
            "selection": "proportionate",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": metric,
            "verbose": False,
            "mixed": False,
            "encode": False,
            "mixed_percentage": 0.9

    }

    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    print("Conf Size: ",len(params))
    print(data)
    for data_i in data:
        try:
            result = genetic_score_comparison(base_path=base_path,
                                              datasets=[data_i],
                                              n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              seed=seed,
                                              metric = metric,
                                              params=params,
                                              n_intervals=n_intervals,
                                              send_email=send_email_cond,
                                              version=3,
                                              email_data={**email_data,
                                                          **{
                                                              "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                              "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}       
                                                        })  
            result.to_csv(
                f"final_result/genetic_3/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")


def execute_ranker_1(data):
    print("RANKER 1")
    grid = {
        "strategy": ["eager","skip"],
        "block_size": [1,2,5,7,10],
        "max_features": [40],
        "max_iterations":[10,15],
        "use_initials": [True]

    }
    
    def_params = {
        "strategy": "skip", 
        "block_size": 10,
        "max_features": 40,
        "verbose": 0,
        "max_err": 0
        }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    
    # params = [
    #     {"strategy": "eager", "block_size": 1, "verbose": 0, "max_err": 0, },
    #     {"strategy": "eager", "block_size": 2, "verbose": 0, "max_err": 0, },
    #     {"strategy": "eager", "block_size": 5, "verbose": 0, "max_err": 0, },
    #     {"strategy": "eager", "block_size": 10, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 1,
    #         "max_iterations": 10, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 2,
    #         "max_iterations": 10, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 5,
    #         "max_iterations": 10, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 10,
    #         "max_iterations": 10, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 1,
    #         "max_features": 40, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 2,
    #         "max_features": 40, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 5,
    #         "max_features": 40, "verbose": 0, "max_err": 0, },
    #     {"strategy": "skip", "block_size": 10,
    #         "max_features": 40, "verbose": 0, "max_err": 0}
    # ]

    for data_i in data[::-1]:
        try:
            result = ranker_score_comparison(base_path=base_path,
                                             datasets=[data_i],
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             seed=seed,

                                             params=params,
                                             n_intervals=n_intervals,
                                             metric=metric,
                                             send_email=send_email_cond,
                                             share_rank=True,
                                             email_data={**email_data,
                                                         **{
                                                             "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                             "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}                                                         })
            result.to_csv(
                f"final_result/ranker_1/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")


def execute_ranker_2(data):
    print("RANKER 2 - prune 3")
    grid = {
        "strategy": ["eager","skip"],
        "block_size": [1,2,5,7,10],
        "max_features": [40],
        "max_iterations":[10,15],
        "prune":[3],

    }
    
    def_params = {
        "strategy": "skip", 
        "block_size": 10,
        "max_features": 40,
        "verbose": 0,
        "max_err": 0
        }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    

    for data_i in data[::1]:
        try:
            result = ranker_score_comparison(base_path=base_path,
                                             datasets=[data_i],
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             seed=seed,
                                             params=params,
                                             n_intervals=n_intervals,
                                             metric=metric,
                                             send_email=send_email_cond,
                                             share_rank=True,
                                             email_data={**email_data,
                                                         **{
                                                             "TITLE": f"{data_i[0]}_{filename_suffix}_PRUNE_3",
                                                             "FILENAME": f"{data_i[0]}_{filename_suffix}_PRUNE_3.csv"}                                                         })
            result.to_csv(
                f"final_result/ranker_2/{data_i[0]}_{filename_suffix}_PRUNE_3.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")

def execute_ranker_3(data):
    print("RANKER 3 - prune 5")
    grid = {
        "strategy": ["eager","skip"],
        "block_size": [1,2,5,7,10],
        "max_features": [40],
        "max_iterations":[10,15],
        "prune":[5],

    }
    
    def_params = {
        "strategy": "skip", 
        "block_size": 10,
        "max_features": 40,
        "verbose": 0,
        "max_err": 0
        }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    

    for data_i in data[::1]:
        try:
            result = ranker_score_comparison(base_path=base_path,
                                             datasets=[data_i],
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             seed=seed,
                                             params=params,
                                             n_intervals=n_intervals,
                                             metric=metric,
                                             send_email=send_email_cond,
                                             share_rank=True,
                                             email_data={**email_data,
                                                         **{
                                                             "TITLE": f"{data_i[0]}_{filename_suffix}_PRUNE_5",
                                                             "FILENAME": f"{data_i[0]}_{filename_suffix}_PRUNE_5.csv"}                                                         })
            result.to_csv(
                f"final_result/ranker_3/{data_i[0]}_{filename_suffix}_PRUNE_5.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")
def execute_ranker_5(data):
    print("RANKER 5 - prune 1")
    grid = {
        "strategy": ["eager","skip"],
        "block_size": [1,2,5,7,10],
        "max_features": [40],
        "max_iterations":[10,15],
        "prune":[1],
        "use_graph":[True]

    }
    
    def_params = {
        "strategy": "skip", 
        "block_size": 10,
        "max_features": 40,
        "verbose": 0,
        "max_err": 0
        }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    

    for data_i in data[::1]:
        try:
            result = ranker_score_comparison(base_path=base_path,
                                             datasets=[data_i],
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             seed=seed,
                                             params=params,
                                             n_intervals=n_intervals,
                                             metric=metric,
                                             send_email=send_email_cond,
                                             share_rank=True,
                                             email_data={**email_data,
                                                         **{
                                                             "TITLE": f"{data_i[0]}_{filename_suffix}_PRUNE_5",
                                                             "FILENAME": f"{data_i[0]}_{filename_suffix}_PRUNE_5.csv"}                                                         })
            result.to_csv(
                f"final_result/ranker_5/{data_i[0]}_{filename_suffix}_PRUNE_1.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")

def execute_ranker_4(data):
    print("RANKER 4 - prune 3 USE INITIALS")
    grid = {
        "strategy": ["eager","skip"],
        "block_size": [1,2,5,7,10],
        "max_features": [40],
        "max_iterations":[10,15],
        "prune":[3],
        "use_initials":[True]

    }
    
    def_params = {
        "strategy": "skip", 
        "block_size": 10,
        "max_features": 40,
        "verbose": 0,
        "max_err": 0
        }
    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val
    
    
    for param in params:
        param["use_initials"] = True

    for data_i in data[::1]:
        try:
            result = ranker_score_comparison(base_path=base_path,
                                             datasets=[data_i],
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             seed=seed,
                                             params=params,
                                             n_intervals=n_intervals,
                                             metric=metric,
                                             send_email=send_email_cond,
                                             share_rank=True,
                                             email_data={**email_data,
                                                         **{
                                                             "TITLE": f"{data_i[0]}_{filename_suffix}_PRUNE_3_INITIALS",
                                                             "FILENAME": f"{data_i[0]}_{filename_suffix}_PRUNE_3_INITIALS.csv"}                                                         })
            result.to_csv(
                f"final_result/ranker_4/{data_i[0]}_{filename_suffix}_PRUNE_3_INITIALS.csv", index=False)
        except Exception as e:
            print(f"Error in database {data_i[0]}: {str(e)}")


def execute_aco_1(data):
    print("ACO1")

    def_params = {
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0.2,
            "beta_evaporation_rate": 0.05,
            "graph_strategy": "mutual_info",
            "use_initials": True,
            "connections": 3,
            "verbose": 0,
            "ants": 5,
            "beta_evaporation_rate": 0.05,
            "iterations": 10,
            "early_stopping": 4,
            "seed": seed,
            "graph_strategy": "mutual_info",
            "update_strategy": "all",
            "final_selection":"BEST",
            "max_errors": 0,
            "save_features": False
        }
    
    grid = {
        "evaporation_rate": [0.05],
        "intensification_factor": [1,2],
        "alpha": [0.2],
        "beta": [0,0.1],
        "use_initials": [False],
        "connections": [1],
        "step":[3],
        "backwards":[False],
        }
    params = []

    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val

    print(f"Configurations: {len(params)}")
    for data_i in data[::1]:
        try:
            result = acfs_score_comparison(base_path=base_path,
                                           datasets=[data_i],
                                           n_splits=n_splits,
                                           n_repeats=n_repeats,
                                           method = 1,
                                           seed=seed,
                                           params=params,
                                           send_email=send_email_cond,
                                           email_data={**email_data,
                                                       **{
                                                           "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                           "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}       
                                                           })
            result.to_csv(f"final_result/aco_3/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"ERROR IN {data_i[0]} DB: {e} ")

def execute_aco_3(data):
    print("ACO3")

    def_params = {
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0.2,
            "beta_evaporation_rate": 0.05,
            "graph_strategy": "mutual_info",
            "use_initials": True,
            "connections": 3,
            "verbose": 0,
            "ants": 1,
            "beta_evaporation_rate": 0.05,
            "iterations": 10,
            "early_stopping": 4,
            "seed": seed,
            "graph_strategy": "mutual_info",
            "update_strategy": "best",
            "final_selection":"BEST",
            "max_errors": 0,
            "save_features": False
        }
    
    grid = {
        "evaporation_rate": [0.05],
        "intensification_factor": [1,2],
        "alpha": [0.2],
        "beta": [0,0.1],
        "use_initials": [False],
        "connections": [1]
        }
    params = []

    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val

    print(f"Configurations: {len(params)}")
    for data_i in data[::1]:
        try:
            result = acfs_score_comparison(base_path=base_path,
                                           datasets=[data_i],
                                           n_splits=n_splits,
                                           n_repeats=n_repeats,
                                           method = 2,
                                           seed=seed,
                                           params=params,
                                           send_email=send_email_cond,
                                           email_data={**email_data,
                                                       **{
                                                           "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                           "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}       
                                                           })
            result.to_csv(f"final_result/aco_3/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"ERROR IN {data_i[0]} DB: {e} ")


def execute_aco_2(data):
    print("ACO")

    def_params = {
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0.2,
            "beta_evaporation_rate": 0.05,
            "early_stopping": 3,
            "graph_strategy": "mutual_info",
            "use_initials": True,
            "connections": 3,
            "verbose": 0,
            "ants": 10,
            "beta_evaporation_rate": 0.05,
            "iterations": 10,
            "early_stopping": 4,
            "seed": seed,
            "graph_strategy": "mutual_info",
            "update_strategy": "all",
            "max_errors": 0,
            "save_features": False
        }
    
    grid = {
        "evaporation_rate": [0.05],
        "intensification_factor": [1,2],
        "alpha": [0.2],
        "beta": [0,0.1,0.2],
        "use_initials": [False],
        "connections": [4],
        "backwards":[False]
        }
    params = []

    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val

    print(f"Configurations: {len(params)}")

    data = [
        ["lenses", "ContactLens"],
        ["abalone", "Rings"],
        ["anneal", "label"],
        ["audiology", "label"],
        ["balance-scale", "label"],
        ["breast-cancer", "Class"],
        ["car-evaluation", "safety"],
        ["derm", "class"],
        ["electricgrid", "stabf"],
        ["glass", "Type"],
        ["horse-colic", "surgery"],
        ["iris", "Species"],
        ["krkp", "label"],
        ["mammographicmasses", "Label"],
        ["mushroom", "class"],
        ["pima", "Outcome"],
        ["student", "Walc"],
        ["voting", "Class Name"],
        ["wine", "class"],
        ["wisconsin", "diagnosis"],
        ["yeast", "nuc"],
        ["tictactoe", "class"],
        ["spam", "class"]
    ]
    data = np.array_split(data, n_computers)[computer-1]
    print(data)
    for data_i in data[::1]:
        try:
            result = acfs_score_comparison(base_path=base_path,
                                           datasets=[data_i],
                                           n_splits=n_splits,
                                           n_repeats=n_repeats,
                                           seed=seed,
                                           params=params,
                                           send_email=send_email_cond,
                                           email_data={**email_data,
                                                       **{
                                                           "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                           "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}       
                                                           })
            result.to_csv(f"final_result/aco_2/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"ERROR IN {data_i[0]} DB: {e} ")


def execute_aco_1(data):
    print("ACO1")

    def_params = {
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0.2,
            "beta_evaporation_rate": 0.05,
            "early_stopping": 3,
            "graph_strategy": "mutual_info",
            "use_initials": True,
            "connections": 3,
            "verbose": 0,
            "ants": 10,
            "beta_evaporation_rate": 0.05,
            "iterations": 10,
            "early_stopping": 4,
            "seed": seed,
            "graph_strategy": "mutual_info",
            "update_strategy": "all",
            "max_errors": 0,
            "save_features": False
        }
    
    grid = {
        "evaporation_rate": [0.05],
        "intensification_factor": [1,2],
        "alpha": [0.2],
        "beta": [0,0.1,0.2],
        "use_initials": [False],
        "connections": [1],
        "backwards":[False]
        }
    params = []

    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key,val in conf.items():
            params[-1][key] = val

    print(f"Configurations: {len(params)}")

    data = np.array_split(data, n_computers)[computer-1]
    print(data)
    for data_i in data[::1]:
        try:
            result = acfs_score_comparison(base_path=base_path,
                                           datasets=[data_i],
                                           n_splits=n_splits,
                                           n_repeats=n_repeats,
                                           seed=seed,
                                           params=params,
                                           send_email=send_email_cond,
                                           email_data={**email_data,
                                                       **{
                                                           "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                           "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}       
                                                           })
            result.to_csv(f"final_result/aco_1/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            print(f"ERROR IN {data_i[0]} DB: {e} ")


setup()
if send_email_cond:
    verify_email()
print("\n------------------------Executing script --------------------------------\n")
execute()
print("\n------------------------ Finished --------------------------------\n")
finish()
