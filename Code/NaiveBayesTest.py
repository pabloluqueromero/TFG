import numpy as np
from sklearn.naive_bayes import GaussianNB,BernoulliNB,CategoricalNB
from NaiveBayesPandas import NaiveBayes as PandasNaiveBayes
from NaiveBayes import NaiveBayes as CustomNaiveBayes
from time import time
import random

if __name__ == "__main__":

    N = 10000
    M = 4
    seed=30
    # Prepare data
    random.seed(seed)
    np.random.seed(seed)

    def generate_data(n, m, rx, ry=3):
        X = np.random.randint(rx, size=(n, m))
        y =  np.random.randint(ry, size=(n))
        return X, y

    for i in range(10):
        print()
        N+=i*10
        M+=i*2
        print(f"Rows: {N}  Features: {M}")
        X, y = generate_data(N, M, 3, 3)
        nbp = PandasNaiveBayes(attributes=list(map(str,range(M))), class_to_predict="C")
        nb_classifier=CustomNaiveBayes(encode_data=False)
        gnb=GaussianNB()
        bnb=BernoulliNB()
        cnb=CategoricalNB()
        ts=time()
        gnb.fit(X, y)
        print(f"GaussianNB {gnb.score(X,y)}  -> Time: {time()-ts}")
        ts=time()
        bnb.fit(X, y)
        print(f"BernouilliNB {bnb.score(X,y)}  -> Time: {time()-ts}")
        ts=time()
        cnb.fit(X, y)
        print(f"CategoricalNB {cnb.score(X,y)}  -> Time: {time()-ts}")
        ts=time()
        # X=np.array([
        #     ['A','-'],
        #     ['B','+'],
        #     ['B','+'],
        #     ['B','+'],
        #     ['B','+'],
        #     ['A','+'],
        # ])
        # y=np.array(['P','O','O','O','O','P'])
        nb_classifier.fit(X,y)
        print(f"CustomNB {nb_classifier.score(X,y)}  -> Time: {time()-ts}")
        ts=time()
        nbp.fit(X,y)
        print(f"CustomNB2 {nbp.score(X,y)}  -> Time: {time()-ts}")

