import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from time import time

#Local imports
from tfg.encoder import CustomOrdinalFeatureEncoder


def test_custom_ordinal_time_comparison(X=None,iterations=10,verbose=1):
    if not X:
        X = np.array([
            ["P","+"],
            ["P2","-"],
            ["P3","-"],
        ])

    custom_encoder = CustomOrdinalFeatureEncoder()
    ordinal_encoder = OrdinalEncoder()

    ordinal_encoder_time =[]
    custom_encoder_time = []
    for i in range(iterations):
        ts=time()
        custom_encoder.fit(X)
        transformed = custom_encoder.transform(X)
        custom_encoder.inverse_transform(transformed)
        custom_encoder_time.append(time()-ts)
        
        ts=time()
        ordinal_encoder.fit(X)
        transformed = ordinal_encoder.transform(X)
        ordinal_encoder.inverse_transform(transformed)
        ordinal_encoder_time.append(time()-ts)
    custom_encoder_time = np.mean(custom_encoder_time)
    ordinal_encoder_time = np.mean(ordinal_encoder_time)
    if verbose:
        print(f"CustomEncoder -> Time: {custom_encoder_time}")
        print(f"OrdinalEncoder -> Time: {ordinal_encoder_time}")
    return custom_encoder_time, ordinal_encoder_time
        
            