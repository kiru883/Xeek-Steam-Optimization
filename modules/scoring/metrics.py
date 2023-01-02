import numpy as np

from sklearn.metrics import mean_squared_error


def RMSE(estimator, X, y):
    y_pred = estimator.predict(X)
    score = mean_squared_error(y, y_pred, squared=False)

    #print(y_pred)
    #print(y)
    return score


def RMSE_satorig_transformed(estimator, X, y, orig_sat):
    y_pred = estimator.predict(X)
    # transform from exp 1.6
    y_pred = orig_sat - y_pred.reshape(-1, 1)
    print(y.shape)
    print(y_pred.shape)
    score = mean_squared_error(y, y_pred, squared=False)
    return score


def RMSE_log_transformed(estimator, X, y):
    y_pred = estimator.predict(X)
    # decode log transform
    eyp = np.e ** y_pred
    y_pred = eyp / (1 + eyp)

    score = mean_squared_error(y, y_pred, squared=False)
    return score