import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tfg.wrapper import PazzaniWrapper

def twospirals(n_points, noise=.5):
    """
    Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))).astype(int)) 

def test_pazzani_wrapper_bsej():
    seed = 200
    X,y = make_classification(n_samples=100, 
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
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, 
                                        test_size=0.3, 
                                        random_state=seed,
                                        stratify=y)
    pw = PazzaniWrapper(seed,strategy="BSEJ",verbose=2)
    transformer,features,model = pw.search(X_train,y_train)

def test_pazzani_wrapper_fssj():
    seed = 200
    X,y = make_classification(n_samples=100, 
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
    # np.random.seed(200)
    # X,y = twospirals(50000)
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, 
                                        test_size=0.3, 
                                        random_state=seed,
                                        stratify=y)
    pw = PazzaniWrapper(seed,strategy="FSSJ",verbose=2)
    transformer,features,model = pw.search(X_train,y_train)


def xor_test():
    pass