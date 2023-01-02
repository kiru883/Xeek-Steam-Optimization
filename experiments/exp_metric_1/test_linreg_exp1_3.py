import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from modules.processing.experiment_preprocessings import preproc_1_1
from modules.scoring.metrics import RMSE
from modules.scoring.cross_val_score import get_cv_rmse_score_1


if __name__ == '__main__':
    # model params
    CV = 4

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

    norm = Normalizer()
    df = pd.read_csv(train_df_path)
    preproc_df = preproc_1_1(df, columns=needest_columns + [target_col] + [group_col])
    preproc_df[needest_columns] = norm.fit_transform(preproc_df[needest_columns].to_numpy())


    # test model
    linreg = LinearRegression()
    scores, _ = get_cv_rmse_score_1(linreg, preproc_df,
                                    x_cols=needest_columns,
                                    y_col=target_col,
                                    group_col=group_col,
                                    cv=CV)
    print(f'RMSE: {scores}')
    print(f'RMSE mean: {sum(scores) / CV}')

