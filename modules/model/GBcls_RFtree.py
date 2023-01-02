import numpy as np
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import *
from sklearn.preprocessing import Normalizer


class GBcls_RFtree():
    def __init__(self, n_estimators=3000, criterion='squared_error', n_jobs=-1, max_features=1.0, random_state=42):
        self.__rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs,
                                        criterion=criterion, max_features=max_features)
        self.__gb_tree = lgb.LGBMClassifier(n_estimators=400)


    def fit(self, X, y):
        cls_y = self.__get_cls_target(y)
        self.__gb_tree.fit(X, cls_y)

        new_X = np.concatenate([X, self.__gb_tree.predict_proba(X)], axis=1)
        self.__rf.fit(new_X, y)
        return self


    def predict(self, X):
        new_X = np.concatenate([X, self.__gb_tree.predict_proba(X)], axis=1)

        tst = self.__rf.predict(new_X)
        return tst

    def __get_cls_target(self, y):
        new_y = y.copy()
        new_y[new_y != 1] = 0
        new_y = new_y.astype(int)

        return new_y