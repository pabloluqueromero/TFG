import subprocess
import argparse
import numpy as np
import itertools

from time import sleep
from tfg.executions import *
from tfg.utils._utils import get_X_y_from_database
from tfg.utils import EmailSendCSV
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

parser.add_argument("--email", required=False, help="email")
parser.add_argument("--password", required=False, help="password")
parser.add_argument("--n_computers", default=1, help="")
parser.add_argument("--computer", default=1, help="computer/n_computers")
parser.add_argument("--metric", default="accuracy", help="scorer")
parser.add_argument("--filename", default="", help="suffix for the")
parser.add_argument("--no_email", action="store_true", help="dont_send_email")
parser.add_argument("--numerical", action="store_true", help="dont_send_email")

args = parser.parse_args()
email = args.email
password = args.password
n_computers = int(args.n_computers)
computer = int(args.computer)-1
metric = args.metric
send_email_cond = not args.no_email
filter_numeric = args.numerical
filename_suffix = args.filename


if filter_numeric:
    filtered_data = []
    for name, label in datasets:
        test = f"{name}.test.csv"
        data = f"{name}.data.csv"
        X, _ = get_X_y_from_database(base_path, name, data, test, label)
        if X.select_dtypes("float").shape[1] > 0:
            filtered_data.append((name, label))
    datasets = filtered_data


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


###################################################################

def execute_commands(commands):
    for command in commands:
        print("\nExecuting: - {"+command+"}")
        subprocess.call(command)
        sleep(3)


def setup():
    print("\n------------------------Setting up environment--------------------------------\n")

def verify_email():

    print("Verifying email . . . . . . . .", end=" ")
    EmailSendCSV(email, email, password).send_test()
    print("OK!")


def finish():
    print("\n------------------------Removing environment--------------------------------\n")
    commands = [
        'history -c',
        'shutdown now'
    ]

    execute_commands(commands)


def execute():
    datas = np.array_split(datasets, n_computers)
    data = datas[computer-1]

    for d in data:
        print(d)

    execute_genetic(data)



def execute_genetic(data):
    print("GENETIC 3")
    grid = {
        "mutation_probability": [0,0.05],
        "selection": ["rank"],
        "combine": ["elitism"],
        "mutation": ["simple"],
        "mixed": [True]
    }
    def_params = {
        "seed": seed,
        "individuals": 20,
        "generations": 30,
        "mutation_probability": 0.01,
        "selection": "proportionate",
        "combine": "elitism",
        "n_intervals": 5,
        "metric": metric,
        "verbose": False,
        "mixed": False,
        "encode_data": False,
        "mixed_percentage": 0.5

    }

    params = []
    for conf in product_dict(**grid):
        params.append(def_params.copy())
        for key, val in conf.items():
            params[-1][key] = val
    print("Conf Size: ", len(params))
    print(data)
    for data_i in data:
        try:
            result = genetic_score_comparison(base_path=base_path,
                                              datasets=[data_i],
                                              n_splits=n_splits,
                                              n_repeats=n_repeats,
                                              seed=seed,
                                              metric=metric,
                                              params=params,
                                              n_intervals=n_intervals,
                                              send_email=send_email_cond,
                                              email_data={**email_data,
                                                          **{
                                                              "TITLE": f"{data_i[0]}_{filename_suffix}",
                                                              "FILENAME": f"{data_i[0]}_{filename_suffix}.csv"}
                                                          })
            result.to_csv(
                f"final_result/genetic_3/{data_i[0]}_{filename_suffix}.csv", index=False)
        except Exception as e:
            raise e
            print(f"Error in database {data_i[0]}: {str(e)}")

setup()
if send_email_cond:
    verify_email()
print("\n------------------------Executing script --------------------------------\n")
execute()
print("\n------------------------ Finished --------------------------------\n")
finish()
