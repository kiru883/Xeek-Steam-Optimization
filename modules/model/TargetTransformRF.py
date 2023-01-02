import numpy as np
import scipy

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer


class TargetTransformRF():
    def __init__(self, n_estimators=3000, criterion='squared_error', n_jobs=-1, max_features=1.0, random_state=42):
        self.__rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs,
                                          criterion=criterion, max_features=max_features)

    def fit(self, X, y):



        self.__rf.fit(X, y)
        return self

    def predict(self, X):



        tst = self.__rf.predict(X)
        return tst

