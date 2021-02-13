import numpy as np
import pandas as pd

from itertools import product
from sklearn.datasets import make_classification
from sklearn.naive_bayes import CategoricalNB,GaussianNB

from tfg.encoder import CustomOrdinalFeatureEncoder
from tfg.naive_bayes import NaiveBayes
from tfg.utils import make_discrete
from time import time



def update_df(df,
                clf,
                rows,
                columns,
                nb_fit_time,
                nb_predict_time,
                nb_score,
                nb_errors):
    row  =  [clf,
                  rows,
                  columns,
                  np.mean(nb_fit_time),
                  np.std(nb_fit_time),
                  0 if nb_errors else np.mean(nb_predict_time),
                  0 if nb_errors else np.std(nb_predict_time),
                  0 if nb_errors else np.mean(nb_score),
                  nb_errors]
    df.loc[len(df)]=row
    df.to_csv("implementation_comparison.csv")


def scoring_comparison(datasets,labels,filename=None,load_data=False):
    column_names = ["dataset",
                    "custom_training_score",
                    "custom_test_score",
                    "categorical_training_score",
                    "categorical_test_score"]
    data =[]
    clf_no_encoding = NaiveBayes(encode_data=False)
    clf_categorical_sklearn = CategoricalNB()
    for dataset in datasets:
        name,X_train,X_test,y_train,y_test = dataset
        
        clf_no_encoding.fit(X_train,y_train)
        custom_train = clf_no_encoding.score(X_train,y_train)
        custom_test = clf_no_encoding.score(X_test,y_test)
       
        clf_categorical_sklearn.min_categories = [np.unique([X_train[:j],X_test[:,j]]) for j in X_train.shape[:,1]]
        clf_categorical_sklearn.fit(X_train,y_train)
        sklearn_train = clf_categorical_sklearn.score(X_train,y_train)
        sklearn_test = clf_categorical_sklearn.score(X_test,y_test)
        data.append([name,custom_train,custom_test,sklearn_train,sklearn_test])
    return pd.DataFrame(data,columns = column_names)