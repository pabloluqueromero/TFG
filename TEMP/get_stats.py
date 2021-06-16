import os
import pandas as pd


from tfg.utils import get_X_y_from_database
import numpy as np

def get_name(x):
    import re
    return re.sub(r'_.*$', '', x),
database_identif = dict()
ordered_names = [
     "abalone",
    "anneal",
    "audiology",
    "balance-scale",
    "breast-cancer",
    "car-evaluation",
    "cmc",
    "credit",
    "cylinder-bands",
    "derm",
    "electricgrid",
    "glass",
    "horse-colic",
    "iris",
    "krkp",
    "lenses",
    "mammographicmasses",
    "mushroom",
    "pima",
    "spam",
    "student",
    "tictactoe",
    "voting",
    "wine",
    "wisconsin",
    "yeast"
]


for i,name in enumerate(ordered_names,start=1):
    database_identif[name]=f"\#{i}"

def database_appendix(datasets, 
                base_path):
    dataset_tqdm = datasets
    connections_list = [1,3,5,10]
    lines = []
    for database in dataset_tqdm:
        name, label = database
        if os.path.exists(base_path+name):
            test = f"{name}.test.csv"
            data = f"{name}.data.csv"
            X, y = get_X_y_from_database(base_path, name, data, test, label)
        else:
            print(f"{name} doesnt' exist")
        
        identif = database_identif[name]
        n_samples,n_features = X.shape
        classes, n_counts = np.unique(y, return_counts=True)
        n_classes = classes.shape[0]
        n_counts = np.round(np.max(n_counts / n_samples),2)
        

        lines.append([
            identif,
            name,
            n_features,
            n_samples,
            n_classes,
            n_counts
        ])

        # if identif == '\#13':
        #     print(lines[-1])
    result = ""
    for line in sorted(lines, key=lambda x: int(x[0][2:])):
        result += "\hline\n" + (" & ".join(map(str,line))) + "\\\\"
    return result

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


lines =  database_appendix(datasets,base_path)

cut = lines.find("\#14")

print(f'''\\begin{{tabular}}{{|c|c|c|c|c|c|}} % Tres primeras con anchura predefinida
\hline 
\\rowcolor{{tema!10}} % Color con transparencia correspondiente
\\bft{{}}  &  
\\bft{{}}  & 
\\bft{{Number}}&
\\bft{{Number}}  &
\\bft{{Number}} &
\\bft{{Most frequent}} \\\\
\\rowcolor{{tema!10}} % Color con transparencia correspondiente
 \multirow{{-2}}{{*}}{{\\bft{{ID}}}}  & 
 \multirow{{-2}}{{*}}{{\\bft{{Name}}}}  & 
\\bft{{of features}} &
\\bft{{of instances}} &
\\bft{{of classes}} &
\\bft{{class proportion}} \\\\
{lines[:cut]}
\hline
\end{{tabular}}''')
print(f'''\\begin{{tabular}}{{|c|c|c|c|c|c|}} % Tres primeras con anchura predefinida
\hline 
\\bft{{}}  &  
\\bft{{}}  & 
\\bft{{Number}}&
\\bft{{Number}}  &
\\bft{{Number}} &
\\bft{{Most frequent}} \\
\\rowcolor{{tema!10}} % Color con transparencia correspondiente
 \multirow{{-2}}{{*}}{{\\bft{{ID}}}}  & 
 \multirow{{-2}}{{*}}{{\\bft{{Name}}}}  & 
\\bft{{of features}} &
\\bft{{of instances}} &
\\bft{{of classes}} &
\\bft{{class proportion}} \\\\
{lines[cut:]}
\hline
\end{{tabular}}''')