import numpy as np
import random
from sklearn.naive_bayes import GaussianNB,BernoulliNB,CategoricalNB, MultinomialNB 
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from time import time

from tfg.naive_bayes import PandasNaiveBayes
from tfg.naive_bayes import NaiveBayes as CustomNaiveBayes
from tfg.encoder import CustomOrdinalFeatureEncoder




def test_remove_feature():

    X,y = make_classification(n_samples=1000, 
                          n_features=100, 
                          n_informative=2, 
                          n_redundant=0, 
                          n_repeated=0, 
                          n_classes=2, 
                          n_clusters_per_class=1, 
                          weights=None,  
                          class_sep=1.0, 
                          hypercube=True, 
                          scale=2.0, 
                          shuffle=True, 
                          random_state=0)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X,y)
    nb.remove_feature(0)
    independent = nb.indepent_term_
    probabilities = nb.probabilities_
    removed = nb.predict_proba(np.delete(X,0,axis=1))
    nb.fit(np.delete(X,0,axis=1),y)
    og = nb.predict_proba(np.delete(X,0,axis=1))
    assert np.allclose(og,removed)
    assert np.allclose(nb.indepent_term_,independent)
    assert np.allclose(nb.probabilities_,probabilities)


def test_add_features():
    X,y = make_classification(n_samples=1000, 
                          n_features=100, 
                          n_informative=2, 
                          n_redundant=0, 
                          n_repeated=0, 
                          n_classes=2, 
                          n_clusters_per_class=1, 
                          weights=None,  
                          class_sep=1.0, 
                          hypercube=True, 
                          scale=2.0, 
                          shuffle=True, 
                          random_state=0)
    X_two_less = np.delete(X,[0,1],axis=1)
    nb = CustomNaiveBayes(encode_data=True)
    nb.fit(X_two_less,y)
    nb.add_features(X[:,[0,1]],y)
    independent = nb.indepent_term_
    probabilities = nb.probabilities_
    added = nb.predict_proba(X)
    nb.fit(X,y)
    og = nb.predict_proba(X)
    assert np.allclose(og,added)
    assert np.allclose(nb.probabilities_,probabilities)
    assert np.allclose(nb.indepent_term_,independent)
# if __name__ == "__main__":
#     a1 = []
#     a2 = []
#     a3 = []
#     a4 = []
#     a1_score = []
#     a2_score = []
#     a3_score = []
#     def make_discrete(X,m=100):
#         X*=m
#         minimum = np.amin(X)
#         if minimum <0:
#             minimum*=-1
#             X+= minimum
#         return X.astype(int)
#     def twospirals(n_points, noise=.5,seed=0):
#         """
#         Returns the two spirals dataset.
#         """
#         np.random.seed(seed)
#         n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
#         d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
#         d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
#         return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
#                 np.hstack((np.zeros(n_points),np.ones(n_points))))
#     import os
#     for i in range(5):
#         # os.system( 'cls' )
#         print(i)
#         # X,y = twospirals(500000,seed=i)
#         # X,y = make_classification(n_samples=100000, 
#         #                   n_features=200, 
#         #                   n_informative=190, 
#         #                   n_redundant=10, 
#         #                   n_repeated=0, 
#         #                   n_classes=2, 
#         #                   n_clusters_per_class=2, 
#         #                   weights=None,  
#         #                   class_sep=1.0, 
#         #                   hypercube=True, 
#         #                   scale=2.0, 
#         #                   shuffle=True, 
#         #                   random_state=0)
#         X,y = make_classification(n_samples=1000, 
#                           n_features=100, 
#                           n_informative=2, 
#                           n_redundant=0, 
#                           n_repeated=0, 
#                           n_classes=2, 
#                           n_clusters_per_class=1, 
#                           weights=None,  
#                           class_sep=1.0, 
#                           hypercube=True, 
#                           scale=2.0, 
#                           shuffle=True, 
#                           random_state=0)
#         X = make_discrete(X,m=1).astype(int)
#         # X = make_discrete(X,m=10)
#         y = y.astype(int)
#         # X2 = X
#         nbp = PandasNaiveBayes(attributes=list(map(str,range(2))), class_to_predict="C")
#         nb_classifier=CustomNaiveBayes(encode_data=True)
#         nb_classifier2=CustomNaiveBayes(encode_data=True)
#         cnb=CategoricalNB()
#         ts=time()
#         c  = CustomOrdinalFeatureEncoder()
#         X2 = c.fit_transform(X)
#         c.transform(X)
#         cnb.fit(X2, y)
#         # proba1 = cnb.predict_proba(X2)
#         # pred1 = cnb.predict(X2)
#         a = cnb.score(X2,y)
#         a1.append(time()-ts)
#         a1_score.append(a)
#         # print(f"CategoricalNB {cnb.score(X2,y)}  -> Time: {time()-ts}")
#         ts=time()
#         nb_classifier.fit(X,y)
#         # proba2=nb_classifier.predict_proba(X)
#         # pred2=nb_classifier.predict(X)
#         # print(proba1[pred1!=pred2],pred1[pred1!=pred2])
#         # print(proba2[pred1!=pred2],pred2[pred1!=pred2])
#         # print(f"CustomNB {nb_classifier.score(X,y)}  -> Time: {time()-ts}")
#         b = nb_classifier.score(X,y)
#         a2_score.append(b)
#         a2.append(time()-ts)
#         ts=time()
#         # nb_classifier2.fit(X,y)
#         # proba2=nb_classifier.predict_proba(X)
#         # pred2=nb_classifier.predict(X)
#         # print(proba1[pred1!=pred2],pred1[pred1!=pred2])
#         # print(proba2[pred1!=pred2],pred2[pred1!=pred2])
#         # print(f"CustomNB {nb_classifier.score(X,y)}  -> Time: {time()-ts}")
#         # b = nb_classifier2.score(X,y)
#         a = nb_classifier2.leave_one_out_cross_val2(X,y)
#         a3.append(time()-ts)
#         ts=time()
#         score_avg = []
#         score_avg2 = []
#         l = LeaveOneOut()
#         score_avg_proba = []
#         for train_index, test_index in l.split(X):
#             # print("INdices",train_index,test_index)
#             nb_classifier.fit(X[train_index],y[train_index])
#             # cnb.fit(X[train_index],y[train_index])
#             score_avg.append(nb_classifier.predict(X[test_index])[0]==y[test_index])
#             # score_avg2.append(cnb.predict(X[test_index])[0])
#             score_avg_proba.append(nb_classifier.predict_proba(X[test_index])[0])
#         score_avg = np.mean(score_avg)
#         a4.append(time()-ts)
#         if i==0:
#             print(a,score_avg)
#             print("All equal",(a== score_avg).all())
#         # print("Same?: ", (np.array(a) == np.array(score_avg)).sum()/len(a))
#         # print(a[a!=np.array(score_avg)])
#         # print(np.array(score_avg)[a!=np.array(score_avg)])
#         # print(np.array(score_avg_proba)[a!=np.array(score_avg)])
#         # ts=time()
#         # nbp.fit(X,y)
#         # print(f"CustomNB2 {nbp.score(X,y)}  -> Time: {time()-ts}")
        
#     # X,y = twospirals(5000)
#     # X = make_discrete(X,m=100)


#     # print("Categorical: ",np.mean(a1[1:]))
#     # print("Custom: ",np.mean(a2[1:]))
#     # print("CustomLoo: ",np.mean(a3[1:]))
#     # print("Categorical: ",np.mean(a1_score))
#     # print("Custom: ",np.mean(a2_score))
#     print("CustomLoo: ",np.mean(a3[1:]))
#     print("Custom SCIKIT LOO: ",np.mean(a4[1:]))

#     # print(nb_classifier.predict_proba(X)[:2])#[cnb.predict(X2)!=nb_classifier.predict(X)])
#     # print(cnb.predict_proba(X2)[:2])#[cnb.predict(X2)!=nb_classifier.predict(X)])
# # print(cnb.predict_proba(X2))