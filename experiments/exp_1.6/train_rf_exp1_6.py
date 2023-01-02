import pandas as pd
import lightgbm as gbm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor
from sklearn.svm import SVR
from modules.processing.experiment_preprocessings import preproc_1_6_exp, preproc_1_5_exp
from modules.scoring.metrics import RMSE
from modules.scoring.cross_val_score import get_cv_rmse_score_1

if __name__ == '__main__':
    # model params
    random_state = 42
    CV = 4
    n_trees = 3000
    criterion = 'squared_error'

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_save_path = PROJ_PATH + "data/models/exp_1.6/"
    train_df_path = PROJ_PATH + "data/raw/train.csv"

    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    group_col = 'CMPL_FAC_ID'
    target_col = 'PCT_DESAT_TO_ORIG'
    time_col = 'SURV_DTE'

    df = pd.read_csv(train_df_path)

    preproc_df = preproc_1_5_exp(df, base_features=needest_columns + [target_col] + [group_col] + [time_col])
    # get meedest columns
    needest_sand_columns = [c for c in preproc_df.columns if 'SAND' in c]
    needest_columns = needest_columns + needest_sand_columns + ['steam_inj_nums'] +\
                      ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time']

    X, y = preproc_df[[c for c in needest_columns if c != target_col]], preproc_df[target_col]
    print(X.columns)
    X, y = X.to_numpy(), y.to_numpy().reshape(-1, 1)

    # fit, scoring, save
    model = StackingRegressor(
        estimators=[
            ('rf1',
             RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion, n_jobs=-1,
                                   max_features=0.1)),
            ('SVR', SVR()),
            ('gbm', gbm.LGBMRegressor(n_estimators=500))

        ]
    )
    model.fit(X, y)
    joblib.dump(model, model_save_path + f'stacking_ad.joblib')

