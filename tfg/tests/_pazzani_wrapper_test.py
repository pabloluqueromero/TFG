import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Local imports
from tfg.utils import make_discrete, generate_xor_data
from tfg.wrapper import PazzaniWrapper
from tfg.wrapper import PazzaniWrapperNB


def test_fssj_xor_problem():
    X, y = generate_xor_data()

    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=200,
                                        stratify=y,
shuflle=True)

    pw = PazzaniWrapper(200, strategy="FSSJ", verbose=2)
    transformer, features, model = pw.search(X_train, y_train)


def test_pazzani_wrapper_bsej():
    seed = 200
    # X,y = make_classification(n_samples=100,
    #                     n_features=10,
    #                     n_informative=7,
    #                     n_redundant=0,
    #                     n_repeated=0,
    #                     n_classes=2,
    #                     n_clusters_per_class=2,
    #                     weights=None,
    #                     class_sep=1.0,
    #                     hypercube=True,
    #                     scale=2.0,
    #                     shuffle=True,
    #                     random_state=seed)
    X, y = make_classification(
                          n_samples=10000,
                          n_features=10,

#                           n_samples=100,
#                           n_features=500000,
                          n_informative=7,
                          n_redundant=0,
                          n_repeated=0,
                          n_classes=2,
                          n_clusters_per_class=2,
                          weights=None,
                          class_sep=1.0,
                          hypercube=True,
                          scale=2.0,
                          shuffle=True,
                          random_state=seed)
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=seed,
                                        stratify=y,
shuflle=True)
    pw = PazzaniWrapper(seed, strategy="BSEJ", verbose=1)
    transformer, features, model = pw.search(X_train, y_train)


def test_pazzani_wrapper_bsej_nb():
    seed = 200
    # X,y = make_classification(n_samples=100,
    #                     n_features=10,
    #                     n_informative=7,
    #                     n_redundant=0,
    #                     n_repeated=0,
    #                     n_classes=2,
    #                     n_clusters_per_class=2,
    #                     weights=None,
    #                     class_sep=1.0,
    #                     hypercube=True,
    #                     scale=2.0,
    #                     shuffle=True,
    #                     random_state=seed)
    X, y = make_classification(
                          n_samples=10000,
                          n_features=10,

#                           n_samples=100,
#                           n_features=500000,
                          n_informative=7,
                          n_redundant=0,
                          n_repeated=0,
                          n_classes=2,
                          n_clusters_per_class=2,
                          weights=None,
                          class_sep=1.0,
                          hypercube=True,
                          scale=2.0,
                          shuffle=True,
                          random_state=seed)
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=seed,
                                        stratify=y,
shuflle=True)
    X, y = make_classification(n_samples=10000,
                            n_features=10,
                            n_informative=7,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            n_clusters_per_class=2,
                            weights=None,
                            class_sep=1.0,
                            hypercube=True,
                            scale=2.0,
                            shuffle=True,
                            random_state=seed)
    X = make_discrete(X, m=10)

    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=200,
                                        stratify=y,
shuflle=True)
    from time import time
    ts = time()
    pw = PazzaniWrapperNB(200, strategy="BSEJ", verbose=1)
    transformer, features, model = pw.search(X_train, y_train)
    print(f"Seconds: {time()-ts}")


def test_pazzani_wrapper_fssj_nb():
    seed = 200
    # X,y = make_classification(n_samples=100,
    #                     n_features=10,
    #                     n_informative=7,
    #                     n_redundant=0,
    #                     n_repeated=0,
    #                     n_classes=2,
    #                     n_clusters_per_class=2,
    #                     weights=None,
    #                     class_sep=1.0,
    #                     hypercube=True,
    #                     scale=2.0,
    #                     shuffle=True,
    #                     random_state=seed)
    X, y = make_classification(
                          n_samples=10000,
                          n_features=10,

#                           n_samples=100,
#                           n_features=500000,
                          n_informative=7,
                          n_redundant=0,
                          n_repeated=0,
                          n_classes=2,
                          n_clusters_per_class=2,
                          weights=None,
                          class_sep=1.0,
                          hypercube=True,
                          scale=2.0,
                          shuffle=True,
                          random_state=seed)
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=seed,
                                        stratify=y,
                                        shuflle=True)
    X, y = make_classification(n_samples=10000,
                            n_features=10,
                            n_informative=7,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            n_clusters_per_class=2,
                            weights=None,
                            class_sep=1.0,
                            hypercube=True,
                            scale=2.0,
                            shuffle=True,
                            random_state=seed)
    X = make_discrete(X, m=10)

    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.3,
                                        random_state=200,
                                        stratify=y,)
    from time import time
    ts = time()
    pw = PazzaniWrapperNB(200, strategy="FSSJ", verbose=1)
    transformer, features, model = pw.search(X_train, y_train)
    print(f"Seconds: {time()-ts}")


def test_pazzani_wrapper_fssj():


    # seed = 200
    # X,y = make_classification(n_samples=100,
    #                     n_features=10,
    #                     n_informative=7,
    #                     n_redundant=0,
    #                     n_repeated=0,
    #                     n_classes=2,
    #                     n_clusters_per_class=2,
    #                     weights=None,
    #                     class_sep=1.0,
    #                     hypercube=True,
    #                     scale=2.0,
    #                     shuffle=True,
    #                     random_state=seed)
    # # np.random.seed(200)
    # # X,y = twospirals(50000)
    # X_train, X_test, y_train, y_test = train_test_split(
    #                                     X, y,
    #                                     test_size=0.3,
    #                                     random_state=seed,
    #                                     stratify=y,shuflle = True)
    # pw = PazzaniWrapper(seed,strategy="FSSJ",verbose=2)
    # transformer,features,model = pw.search(X_train,y_train)

    seed=200
    # X,y = make_classification(n_samples=100,
    #                     n_features=10,
    #                     n_informative=7,
    #                     n_redundant=0,
    #                     n_repeated=0,
    #                     n_classes=2,
    #                     n_clusters_per_class=2,
    #                     weights=None,
    #                     class_sep=1.0,
    #                     hypercube=True,
    #                     scale=2.0,
    #                     shuffle=True,
    #                     random_state=seed)
    X, y=make_classification(
                          n_samples = 10000,
                          n_features = 10,

#                           n_samples=100,
#                           n_features=500000,
                          n_informative = 7,
                          n_redundant = 0,
                          n_repeated = 0,
                          n_classes = 2,
                          n_clusters_per_class = 2,
                          weights = None,
                          class_sep = 1.0,
                          hypercube = True,
                          scale = 2.0,
                          shuffle = True,
                          random_state = seed)
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test=train_test_split(
                                        X, y,
                                        test_size = 0.3,
                                        random_state = seed,
                                        stratify = y,
shuflle = True)
    X, y=make_classification(n_samples = 10000,
                            n_features = 10,
                            n_informative = 7,
                            n_redundant = 0,
                            n_repeated = 0,
                            n_classes = 2,
                            n_clusters_per_class = 2,
                            weights = None,
                            class_sep = 1.0,
                            hypercube = True,
                            scale = 2.0,
                            shuffle = True,
                            random_state = seed)
    X=make_discrete(X, m = 10)

    X_train, X_test, y_train, y_test=train_test_split(
                                        X, y,
                                        test_size = 0.3,
                                        random_state = 200,
                                        stratify = y,
shuflle = True)
    from time import time
    ts=time()
    pw=PazzaniWrapper(200, strategy = "FSSJ", verbose = 1)
    transformer, features, model=pw.search(X_train, y_train)
    print(f"Seconds: {time()-ts}")

def xor_test():
    pass
