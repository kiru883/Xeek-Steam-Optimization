import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from modules.processing.experiment_preprocessings import preproc_1_1


if __name__ == '__main__':
    # model params
    random_state = 42

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_path = PROJ_PATH + "data/models/exp_1.3/linreg_4models_0.3699993706981101.joblib"
    normalizer_path = PROJ_PATH + "data/models/exp_1.3/normalizer_v1.joblib"
    test_df_path = PROJ_PATH + "data/raw/test_data.csv"
    sub_save_path = PROJ_PATH + "data/processed/1_3_subm_v1.csv"
    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]

    df = pd.read_csv(test_df_path)
    norm = joblib.load(normalizer_path)
    preproc_df = preproc_1_1(df, columns=needest_columns)
    X = preproc_df[needest_columns]
    X = X.to_numpy()
    X = norm.transform(X)

    # fit, scoring, save
    preds = 0
    linreg_4list = joblib.load(model_path)
    for model in linreg_4list:
        preds = preds + model.predict(X)
    preds = preds / len(linreg_4list)
    preds = preds.flatten()

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub.to_csv(sub_save_path, index=False)