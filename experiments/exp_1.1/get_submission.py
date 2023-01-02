import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from modules.processing.experiment_preprocessings import preproc_1_1


if __name__ == '__main__':
    # model params
    random_state = 42

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_path = PROJ_PATH + "data/models/exp_1.1/rt_0.2529767922940465.joblib"
    test_df_path = PROJ_PATH + "data/raw/test_data.csv"
    sub_orig_path = PROJ_PATH + 'data/raw/submission_sample.csv'
    sub_save_path = PROJ_PATH + "data/processed/1_1_subm_v1.csv"
    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]

    df = pd.read_csv(test_df_path)
    df_sub = pd.read_csv(sub_orig_path)
    preproc_df = preproc_1_1(df, columns=needest_columns)
    X = preproc_df[needest_columns]
    X = X.to_numpy()

    # fit, scoring, save
    reg_trees = joblib.load(model_path)
    preds = reg_trees.predict(X)

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub = df_sub.round(2)
    df_sub.to_csv(sub_save_path, index=False)