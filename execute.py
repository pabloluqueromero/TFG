from tfg.executions import *
import subprocess
import argparse
from time import sleep
import numpy as np
# py execute.py --email mail --password pass --algorithm ranker --method 2 --n_computers 3 --computer 1

'''
Global Variables
'''
seed = 200
n_repeats = 5
n_splits = 3
n_intervals = 5
base_path = "./UCIREPO/"
datasets = [
    # ["arrhythmia","A279"],
    ["yeast", "nuc"],
    ["abalone", "Rings"],
    ["mammographicmasses", "Label"],
    ["breast-cancer", "Class"],
    ["anneal", "label"],
    ["audiology", "label"],
    ["balance-scale", "label"],
    ["iris", "Species"],
    ["student", "Walc"],
    ["electricgrid", "stabf"],
    ["horse-colic", "surgery"],
    ["glass", "Type"],
    ["krkp", "label"],
    ["mushroom", "class"],
    ["voting", "Class Name"],
    ["credit", "A16"],
    ["pima", "Outcome"],
    ["wine", "class"],
    ["wisconsin", "diagnosis"],
    ["car-evaluation", "safety"],
    ["lenses", "ContactLens"],
    ["cmc", "Contraceptive"],
    ["cylinder-bands", "band type"],
    ["derm", "class"],
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
parser.add_argument("--no_email", action="store_true", help="dont_send_email")

args = parser.parse_args()
email = args.email
password = args.password
algorithm = args.algorithm
method = int(args.method)
n_computers = int(args.n_computers)
computer = int(args.computer)-1
send_email_cond = not args.no_email

if(computer >= n_computers):
    print("ERROR: computer >= n_computers")
    exit(1)


email_data = {
    "FROM": email,
    "TO": email,
    "PASSWORD": password,
}


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
        'rm -rf *',
        'history -c',
        'shutdown now'
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
        else:
            execute_ranker_3(data)
    elif algorithm == "genetic":
        if method == 1:
            execute_genetic_1(data)
        else:
            execute_genetic_2(data)
    elif algorithm == "aco":
        execute_aco_1(data)


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
            "combine": "elitism",
            "n_intervals": 5,
            "metric": "accuracy",
            "verbose": True,
            "flexible_logic": True,
            "encode": False
        },
        {
            "size": 10,
            "seed": seed,
            "individuals": 20,
            "generations": 20,
            "mutation_probability": 0.4,
            "selection": "simple",
            "combine": "truncation",
            "n_intervals": 5,
            "metric": "accuracy",
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
            "metric": "accuracy",
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
            "metric": "accuracy",
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
            "metric": "accuracy",
            "flexible_logic": True,
            "verbose": True,
            "encode": False
        },
    ]

    for data_i in data[::1]:
        result = genetic_score_comparison(base_path=base_path,
                                          datasets=[data_i],
                                          n_splits=n_splits,
                                          n_repeats=n_repeats,
                                          seed=seed,
                                          params=params,
                                          n_intervals=n_intervals,
                                          metric="accuracy",
                                          send_email=True,
                                          version=2,
                                          email_data={**email_data,
                                                      **{
                                                          "TITLE": f"{data_i[0]}_UCLM_MIXED_0.6_HEURISTIC",
                                                          "FILENAME": f"{data_i[0]}_GENETIC2.CSV"}
                                                      })


def execute_genetic_2(data):
    print("GENETIC 2")
    params = [
        {
            "size": 10,
            "seed": seed,
            "individuals": 50,
            "generations": 20,
            "mutation_probability": 0.4,
            "selection": "simple",
            "combine": "truncation",
            "n_intervals": 5,
            "metric": "accuracy",
            "verbose": True
        }, {
            "size": 6,
            "seed": seed,
            "individuals": 50,
            "generations": 20,
            "mutation_probability": 0.2,
            "selection": "rank",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": "accuracy",
            "verbose": True
        }, {
            "size": 3,
            "seed": seed,
            "individuals": 50,
            "generations": 20,
            "mutation_probability": 0.2,
            "selection": "rank",
            "combine": "elitism",
            "n_intervals": 5,
            "metric": "accuracy",
            "verbose": True
        },
    ]

    for data_i in data[::1]:
        genetic_score_comparison(base_path=base_path,
                                 datasets=[data_i],
                                 n_splits=n_splits,
                                 n_repeats=n_repeats,
                                 seed=seed,

                                 params=params,
                                 n_intervals=n_intervals,
                                 metric="accuracy",
                                 send_email=send_email_cond,
                                 version=2,
                                 email_data={**email_data,
                                             **{
                                                 "TITLE": f"{data_i[0]}_UCLM",
                                                 "FILENAME": f"{data_i[0]}_GENETIC2.CSV"}
                                             })


def execute_ranker_1(data):
    print("RANKER 1")
    params = [
        {"strategy": "eager", "block_size": 1, "verbose": 0, "max_err": 0, },
        {"strategy": "eager", "block_size": 2, "verbose": 0, "max_err": 0, },
        {"strategy": "eager", "block_size": 5, "verbose": 0, "max_err": 0, },
        {"strategy": "eager", "block_size": 10, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 1,
            "max_iterations": 10, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 2,
            "max_iterations": 10, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 5,
            "max_iterations": 10, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 10,
            "max_iterations": 10, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 1,
            "max_features": 20, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 2,
            "max_features": 20, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 5,
            "max_features": 20, "verbose": 0, "max_err": 0, },
        {"strategy": "skip", "block_size": 10,
            "max_features": 20, "verbose": 0, "max_err": 0}
    ]

    for data_i in data[::1]:
        result = ranker_score_comparison(base_path=base_path,
                                         datasets=[data_i],
                                         n_splits=n_splits,
                                         n_repeats=n_repeats,
                                         seed=seed,

                                         params=params,
                                         n_intervals=n_intervals,
                                         metric="accuracy",
                                         send_email=send_email_cond,
                                         share_rank=True,
                                         email_data={**email_data,
                                                     **{
                                                         "TITLE": f"{data_i[0]}",
                                                         "FILENAME": f"{data_i[0]}.CSV"}
                                                     })


def execute_ranker_2(data):
    print("RANKER 2")
    params = [
        {"strategy": "eager", "block_size": 1,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 2,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 5,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 1, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 2, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 5, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 10, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 1, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 2, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 5, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 10, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3}
    ]

    for data_i in data[::1]:
        result = ranker_score_comparison(base_path=base_path,
                                         datasets=[data_i],
                                         n_splits=n_splits,
                                         n_repeats=n_repeats,
                                         seed=seed,

                                         params=params,
                                         n_intervals=n_intervals,
                                         metric="accuracy",
                                         send_email=send_email_cond,
                                         share_rank=True,
                                         email_data={**email_data,
                                                     **{
                                                         "TITLE": f"{data_i[0]}",
                                                         "FILENAME": f"{data_i[0]}.CSV"}
                                                     })


def execute_ranker_3(data):
    print("RANKER 3")
    params = [
        {"strategy": "eager", "block_size": 1,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 2,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 5,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "eager", "block_size": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 1, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 2, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 5, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 10, "max_iterations": 10,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 1, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 2, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 5, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3},
        {"strategy": "skip", "block_size": 10, "max_features": 20,
            "verbose": 0, "max_err": 0, "prune": 3}
    ]

    for param in params:
        param["use_initials"] = True

    for data_i in data[::1]:
        result = ranker_score_comparison(base_path=base_path,
                                         datasets=[data_i],
                                         n_splits=n_splits,
                                         n_repeats=n_repeats,
                                         seed=seed,

                                         params=params,
                                         n_intervals=n_intervals,
                                         metric="accuracy",
                                         send_email=send_email_cond,
                                         share_rank=True,
                                         email_data={**email_data,
                                                     **{
                                                         "TITLE": f"{data_i[0]}",
                                                         "FILENAME": f"{data_i[0]}.CSV"}
                                                     })


def execute_aco_1(data):
    print("ACO")
    params = [
        {
            "ants": 10,
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0.5,
            "beta_evaporation_rate": 0.05,
            "iterations": 100,
            "early_stopping": 3,
            "seed": 3,
            "parallel": False,
            "save_features": False,
            "verbose": 0,
            "graph_strategy": "full",
            "connections": 3,
            "update_strategy": "all"
        }, {
            "ants": 10,
            "evaporation_rate": 0.1,
            "intensification_factor": 2,
            "alpha": 0.5,
            "beta": 0,
            "beta_evaporation_rate": 0.05,
            "iterations": 100,
            "early_stopping": 3,
            "seed": 3,
            "parallel": False,
            "save_features": False,
            "verbose": 0,
            "graph_strategy": "full",
            "connections": 3,
            "update_strategy": "all"
        }]

    for data_i in data[::1]:
        acfs_score_comparison(base_path=base_path,
                              datasets=[data_i],
                              n_splits=n_splits,
                              n_repeats=n_repeats,
                              seed=seed,

                              params=params,
                              send_email=send_email_cond,
                              share_rank=True,
                              email_data={**email_data,
                                          **{
                                              "TITLE": f"{data_i[0]}",
                                              "FILENAME": f"{data_i[0]}.CSV"
                                          }})


setup()
if send_email_cond:
    verify_email()
print("\n------------------------Executing script --------------------------------\n")
execute()
print("\n------------------------ Finished --------------------------------\n")
finish()
