import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from modules.scoring.metrics import RMSE, RMSE_satorig_transformed
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture


def get_cv_rmse_score_1(estimator, df, x_cols, y_col, group_col, cv=4, verbosity=True):
    X, y, group = df[x_cols], df[y_col], df[group_col]
    if verbosity:
        print(f'Columns: {X.columns}')
    X, y, group = X.to_numpy(), y.to_numpy().reshape(-1, 1), group.to_numpy().reshape(-1, 1)

    scores = []
    models = []
    gkf = GroupKFold(n_splits=cv)
    for train_index, test_index in gkf.split(X, y, groups=group):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        _fitted = estimator.fit(X_train, y_train)

        #print(X_test[y_test.flatten() == 0][:, 4])
        #print(_fitted.predict(X_test).flatten()[X_test[:, 4].flatten() >= 0.6])
        #print([y_test.flatten() == 0])

        score = RMSE(_fitted, X_test, y_test)
        scores.append(score)
        models.append(_fitted)

    return scores, models


def get_cv_rmse_score_1_test(estimator, df, x_cols, y_col, group_col, cv=4, verbosity=True):
    X, y, group = df[x_cols], df[y_col], df[group_col]
    if verbosity:
        print(f'Columns: {X.columns}')
    X, y, group = X.to_numpy(), y.to_numpy().reshape(-1, 1), group.to_numpy().reshape(-1, 1)

    scores = []
    models = []
    gkf = GroupKFold(n_splits=cv)
    for train_index, test_index in gkf.split(X, y, groups=group):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        _fitted = estimator.fit(X_train, y_train)



        score = RMSE(_fitted, X_test, y_test)
        scores.append(score)
        models.append(_fitted)

    return scores, models