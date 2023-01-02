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
    test_size = 0.2
    n_trees = 3000
    criterion = 'squared_error'

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    test_df_path = PROJ_PATH + "data/raw/test_data.csv"
    model_path = PROJ_PATH + "data/models/exp_1.4/rt_0.24548837129258227.joblib"
    sub_save_path = PROJ_PATH + "data/processed/1_4_subm_v1.csv"
    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]

    df = pd.read_csv(test_df_path)

    # try steam inj. features
    preproc_df = preproc_1_4_exp(df, base_features=needest_columns, use_sand_feature=True,
                                 use_steam_inj_features=True, use_move_sinj_move=True)
    needest_sand_columns4 = [c for c in preproc_df.columns if 'SAND' in c]
    needest_columns4 = needest_columns + needest_sand_columns4 + ['steam_inj_nums']
    X = preproc_df[needest_columns4]
    print(X.columns)
    X = X.to_numpy()

    # fit, scoring, save
    reg_trees = joblib.load(model_path)
    preds = reg_trees.predict(X)

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub.to_csv(sub_save_path, index=False)



