



import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomLabelEncoder, CustomOrdinalFeatureEncoder
from tfg.feature_construction import DummyFeatureConstructor
from tfg.naive_bayes import NaiveBayes
from tfg.genetic_algorithm import GeneticAlgorithm,GeneticAlgorithmV2
from tfg.utils import get_X_y_from_database


def genetic_score_comparison(datasets, seed, test_size, base_path, params, n_iterations=30,n_intervals=5,metric="accuracy",send_email=False,email_data = dict(),share_rank=True,version=1):
    result = []
    dataset_tqdm = tqdm(datasets)

    # Instantiate ranker
    r = GeneticAlgorithm(seed=200) if version==1 else GeneticAlgorithmV2(seed=200)
    nb = NaiveBayes(encode_data=True,n_intervals = n_intervals, metric=metric)
    for database in dataset_tqdm:
        name, label = database
        if os.path.exists(base_path+name):
            test = f"{name}.test.csv"
            data = f"{name}.data.csv"
            X, y = get_X_y_from_database(base_path, name, data, test, label)

            dataset_tqdm.set_postfix({"DATABASE": name})

            seed_tqdm = tqdm(range(n_iterations), leave=False)

            # Set up data structures to store results
            nb_score = np.zeros(shape=(len(params), n_iterations))
            r_score = np.zeros(shape=(len(params), n_iterations))
            r_selected = np.zeros(shape=(len(params), n_iterations))
            r_dummy = np.zeros(shape=(len(params), n_iterations))

            for i in seed_tqdm:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=seed+i,
                        stratify=y,
                        shuffle=True)
                except:
                    #Not enough values to stratify y
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=seed+i,
                        shuffle=True)

                nb.fit(X=X_train, y=y_train)
                naive_bayes_score = nb.score(X_test, y_test)
                c = CustomOrdinalFeatureEncoder(n_intervals = n_intervals)
                X_train = c.fit_transform(X_train)
                X_test = c.transform(X_test)
                l = CustomLabelEncoder()
                y_train = l.fit_transform(y_train)
                y_test = l.transform(y_test)

                conf_index = 0
                for conf in params:
                    seed_tqdm.set_postfix({"seed": i, "config": conf_index})
                    r.set_params(**conf)
                    r.fit(X_train, y_train)
                    
                    # score
                    ranker_score = r.score(X_test, y_test)

                    # Get data
                    n_original_features = len(list(filter(lambda x: isinstance(
                        x, DummyFeatureConstructor), r.best_features)))
                    n_selected = len(r.best_features)

                    # Update
                    nb_score[conf_index, i] = naive_bayes_score
                    r_score[conf_index, i] = ranker_score
                    r_selected[conf_index, i] = n_selected
                    r_dummy[conf_index, i] = n_original_features

                    conf_index += 1

            # Insert to final result averaged metrics for this database
            for conf_index, conf in enumerate(params):
                row = [name,
                       X.shape[1],
                       np.mean(nb_score[conf_index]),
                       np.std(nb_score[conf_index]),
                       np.mean(r_score[conf_index]),
                       np.std(r_score[conf_index]),
                       conf,
                       np.mean(r_selected[conf_index]),
                       np.mean(r_dummy[conf_index])]
                result.append(row)

        else:
            print(f"{name} doesnt' exist")
    columns = ["Database", "Number of attributes", "NBScore","NBScore STD", "Genetic Score","Genetic Score STD",
               "Configuration", "Selected_attributes", "Original"]
    result = pd.DataFrame(result, columns=columns)
    if send_email:
        from tfg.utils import send_results
        send_results("GENETIC",email_data,result)
    return result




base_path = "./UCIREPO/"
data = [
    #["abalone","Rings"],
      # ["adult","income"],
    #["anneal","label"],
    #["audiology","label"],
    #["balance-scale","label"],
    # ["krkopt","Optimal depth-of-win for White"],
    #["iris","Species"],
    #["student","Walc"],
    #["electricgrid","stabf"],
    #["horse-colic","surgery"],
    # #["glass","Type"],
    # ["krkp","label"],
    # ["mushroom","class"],
    # ["voting","Class Name"],
    # ["credit","A16"],
    # ["pima","Outcome"],
    # ["wine","class"],
    # ["wisconsin","diagnosis"],
    # ["car-evaluation","safety"],
    # # ["connect-4","class"],
    ["lenses","ContactLens"],
    # ["cmc","Contraceptive"],
    # ["cylinder-bands","band type"],
    # ["derm","class"],
    # ["tictactoe","class"],
    # ["spam","class"]
]

graphs_folder = "out/graphs/"
csv_folder = "out/csv/"

#create directories
import os 
for directory in [graphs_folder,csv_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

n_iterations=15
seed=200
test_size=0.3
n_intervals=5
params = [
   {"size":10, 
    "seed":seed, 
    "individuals":50, 
    "generations":20,
    "mutation_probability":0.4, 
    "selection":"rank", 
    "combine":"truncation",
    "n_intervals": 5,
    "metric": "accuracy",
    "use_initials":True,
    "verbose":True}
]

email_data = {
    "FROM":"pablocsvtfg@gmail.com",
    "TO":"pablocsvtfg@gmail.com",
    "PASSWORD":"TFGPablo2",

}

n_iterations=30
seed=200
test_size=0.3
for data_i in data:
  result = genetic_score_comparison(base_path=base_path,
                                  datasets=[data_i],
                                  test_size=test_size,
                                  seed=seed,
                                  n_iterations=n_iterations,
                                  params = params,
                                  n_intervals=n_intervals,
                                   metric="accuracy",
                                  send_email=True,
                                  version=2,
                                  email_data = {**email_data,
                                                **{
                                                "TITLE":f"{data_i[0]}",
                                                "FILENAME": f"{data_i[0]}.CSV"}
                                                })
  result.to_csv(csv_folder+f"genetic_score_comparison_5_{data_i[0]}.csv",index=False)