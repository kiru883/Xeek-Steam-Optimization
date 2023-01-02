import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from modules.processing.experiment_preprocessings import preproc_1_1
from modules.scoring.metrics import RMSE
from modules.scoring.cross_val_score import get_cv_rmse_score_1


if __name__ == '__main__':
    # model params
    CV = 4
    random_state = 42
    n_trees = 400
    learning_rate = 0.1

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    train_df_path = PROJ_PATH + "data/raw/train.csv"

    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    group_col = 'CMPL_FAC_ID'
    target_col = 'PCT_DESAT_TO_ORIG'

    df = pd.read_csv(train_df_path)
    preproc_df = preproc_1_1(df, columns=needest_columns + [target_col] + [group_col])


    # test model
    boost_trees = lgb.LGBMRegressor(n_estimators=n_trees, random_state=random_state, learning_rate=learning_rate,
                                  objective='mse')
    scores, _ = get_cv_rmse_score_1(boost_trees, preproc_df,
                                    x_cols=needest_columns,
                                    y_col=target_col,
                                    group_col=group_col,
                                    cv=CV)
    print(f'RMSE: {scores}')
    print(f'RMSE mean: {sum(scores) / CV}')

