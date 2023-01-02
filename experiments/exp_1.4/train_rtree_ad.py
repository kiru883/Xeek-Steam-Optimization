import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from modules.processing.experiment_preprocessings import preproc_1_4_exp
from modules.scoring.metrics import RMSE


if __name__ == '__main__':
    # model params
    random_state = 42
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

    df = pd.read_csv(train_df_path)

    # try steam inj. features
    preproc_df = preproc_1_4_exp(df, base_features=needest_columns + [target_col], use_sand_feature=True,
                                 use_steam_inj_features=True, use_move_sinj_move=True)
    needest_sand_columns4 = [c for c in preproc_df.columns if 'SAND' in c]
    needest_columns4 = needest_columns + needest_sand_columns4 + ['steam_inj_nums']
    X, y = preproc_df[[c for c in needest_columns4 if c != target_col]], preproc_df[target_col]
    print(X.columns)
    X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)

    # fit, scoring, save
    reg_trees = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion)
    reg_trees.fit(X, y)
    joblib.dump(reg_trees, model_save_path + f'rt_ad.joblib')



