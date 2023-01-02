import pandas as pd
import joblib
import lightgbm as lgb

from modules.processing.experiment_preprocessings import preproc_1_1


if __name__ == '__main__':
    # model params
    random_state = 42

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_path = PROJ_PATH + "data/models/exp_1.2/gbt_4models_0.24238347181436584.joblib"
    test_df_path = PROJ_PATH + "data/raw/test_data.csv"
    sub_orig_path = PROJ_PATH + 'data/raw/submission_sample.csv'
    sub_save_path = PROJ_PATH + "data/processed/1_2_subm_v1.csv"
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
    preds = 0
    reg_grad_trees_4list = joblib.load(model_path)
    for model in reg_grad_trees_4list:
        preds = preds + model.predict(X)
    preds = preds / len(reg_grad_trees_4list)

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub.to_csv(sub_save_path, index=False)