import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from modules.processing.experiment_preprocessings import preproc_1_1
from modules.scoring.metrics import RMSE


if __name__ == '__main__':
    # model params
    random_state = 42
    learning_rate = 0.1

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_save_path = PROJ_PATH + "data/models/exp_1.3/"
    normalizer_save_path = PROJ_PATH + "data/models/exp_1.3/"
    train_df_path = PROJ_PATH + "data/raw/train.csv"
    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    target_col = 'PCT_DESAT_TO_ORIG'

    df = pd.read_csv(train_df_path)
    preproc_df = preproc_1_1(df, columns=needest_columns + [target_col])
    X, y = preproc_df[[c for c in needest_columns if c != target_col]], preproc_df[target_col]
    X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)
    # normalize X
    norm = Normalizer()
    X = norm.fit_transform(X)
    # save normalizer
    joblib.dump(norm, normalizer_save_path + 'normalizer_v1.joblib')

    scores = []
    models = []
    kf = KFold(n_splits=4, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # fit, scoring, save
        linreg_model = LinearRegression()
        linreg_model.fit(X_train, y_train)

        score = RMSE(linreg_model, X_test, y_test)
        scores.append(score)
        models.append(linreg_model)

    print(f"RMSE scores: {scores}")
    print(f'RMSE mean score: {sum(scores) / len(scores)}')

    joblib.dump(models, model_save_path + f'linreg_4models_{sum(scores) / len(scores)}.joblib')
