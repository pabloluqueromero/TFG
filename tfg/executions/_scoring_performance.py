import numpy as np
import pandas as pd

from itertools import product
from sklearn.datasets import make_classification
from sklearn.naive_bayes import CategoricalNB,GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
from tqdm.autonotebook import tqdm

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.naive_bayes import NaiveBayes
from tfg.utils import make_discrete
from tfg.utils import get_X_y_from_database


def scoring_comparison(base_path,datasets,verbose=1,test_size=0.3,seed=None,n_iterations=30):
    column_names = ["dataset",
                    "custom_training_score",
                    "custom_test_score",
                    "categorical_training_score",
                    "categorical_test_score"]
    data =[]
    clf_no_encoding = NaiveBayes(encode_data=False)
    clf_categorical_sklearn = CategoricalNB()
    
    datasets_iter = tqdm(datasets, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    c = CustomOrdinalFeatureEncoder()
    l = LabelEncoder()
    for dataset in datasets_iter:
        dataset_name, label = dataset
        data_filename = f"{dataset_name}.data.csv"
        test_filename = f"{dataset_name}.test.csv"
        X, y = get_X_y_from_database(base_path=base_path,
                                     name = dataset_name,
                                     data = data_filename, 
                                     test = test_filename, 
                                     label = label)
        custom_train = []
        custom_test = []

        sklearn_train = []
        sklearn_test = []


        X  = c.fit_transform(X)
        y  = l.fit_transform(y)
        for iteration in range(n_iterations):
            if verbose:
                datasets_iter.set_postfix({"Dataset": dataset_name, "seed":iteration})
                datasets_iter.refresh()
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                             test_size=test_size,
                                                             random_state=seed+iteration,
                                                             shuffle=True,
                                                             stratify=y)

            #Fit
            clf_no_encoding.fit(X_train,y_train)
            clf_categorical_sklearn.min_categories = [1+np.max(np.concatenate([X_train[:,j],X_test[:,j]])) for j in range(X_train.shape[1])]
            clf_categorical_sklearn.fit(X_train,y_train)
            
            #Predict
            custom_train.append(clf_no_encoding.score(X_train,y_train))
            custom_test.append(clf_no_encoding.score(X_test,y_test))
            sklearn_train.append(clf_categorical_sklearn.score(X_train,y_train))
            sklearn_test.append(clf_categorical_sklearn.score(X_test,y_test))
        data.append([dataset_name,np.mean(custom_train),np.mean(custom_test),np.mean(sklearn_train),np.mean(sklearn_test)])
    return pd.DataFrame(data,columns = column_names)