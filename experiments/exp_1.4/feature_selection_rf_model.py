import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from modules.processing.experiment_preprocessings import preproc_1_4_exp
from modules.scoring.metrics import RMSE


def get_cv_rmse_score(estimator, X, y, cv=3, random_state=42):
    scores = []
    kf = KFold(n_splits=cv, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        estimator.fit(X_train, y_train)
        score = RMSE(estimator, X_test, y_test)
        scores.append(score)

    return scores


if __name__ == '__main__':
    # model params
    random_state = 42
    test_size = 0.2
    n_trees = 3000
    criterion = 'squared_error'

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_save_path = PROJ_PATH + "data/models/exp_1.4/"
    train_df_path = PROJ_PATH + "data/raw/train.csv"
    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    target_col = 'PCT_DESAT_TO_ORIG'

    mean_f = lambda x: sum(x) / len(x)

    df = pd.read_csv(train_df_path)

    # try steam inj. num feature
    # needest_columns1 = needest_columns + ['steam_inj_nums']
    # preproc_df = preproc_1_4_exp(df, base_features=needest_columns + [target_col], use_steam_inj_features=True)
    # model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)
    # X, y = preproc_df[[c for c in needest_columns1 if c != target_col]], preproc_df[target_col]
    # print(X.columns)
    # #print(X)
    # X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)
    # scores1 = get_cv_rmse_score(model, X, y, cv=4)
    # print(f'1. RMSE scores: {scores1}')
    # print(f'1. RMSE mean: {mean_f(scores1)}')

    # try steam inj. features
    # preproc_df = preproc_1_4_exp(df, base_features=needest_columns + [target_col], use_move_sinj_move=True)
    # model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)
    # X, y = preproc_df[[c for c in needest_columns if c != target_col]], preproc_df[target_col]
    # print(X.columns)
    # X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)
    # scores2 = get_cv_rmse_score(model, X, y, cv=4)
    # print(f'2. RMSE scores: {scores2}')
    # print(f'2. RMSE mean: {mean_f(scores2)}')

    # try steam inj. features
    # preproc_df = preproc_1_4_exp(df, base_features=needest_columns + [target_col], use_sand_feature=True)
    # model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)
    # needest_sand_columns3 = [c for c in preproc_df.columns if 'SAND' in c]
    # needest_columns3 = needest_columns + needest_sand_columns3
    # X, y = preproc_df[[c for c in needest_columns3 if c != target_col]], preproc_df[target_col]
    # print(X.columns)
    # X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)
    # scores3 = get_cv_rmse_score(model, X, y, cv=4)
    # print(f'3. RMSE scores: {scores3}')
    # print(f'3. RMSE mean: {mean_f(scores3)}')

    # try steam inj. features
    preproc_df = preproc_1_4_exp(df, base_features=needest_columns + [target_col], use_sand_feature=True,
                                 use_steam_inj_features=True, use_move_sinj_move=True)
    model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)
    needest_sand_columns4 = [c for c in preproc_df.columns if 'SAND' in c]
    needest_columns4 = needest_columns + needest_sand_columns4 + ['steam_inj_nums']
    X, y = preproc_df[[c for c in needest_columns4 if c != target_col]], preproc_df[target_col]
    print(X.columns)
    X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)
    scores4 = get_cv_rmse_score(model, X, y, cv=4)
    print(f'4. RMSE scores: {scores4}')
    print(f'4. RMSE mean: {mean_f(scores4)}')
    # fit, scoring, save
    #reg_trees = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)

    #reg_trees.fit(X_train, y_train)
    #score = RMSE(reg_trees, X_test, y_test)
    #print(f'RMSE: {score}')
    #joblib.dump(reg_trees, model_save_path + f'rt_{score}.joblib')



