import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from modules.processing.experiment_preprocessings import preproc_1_7_remake
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
    model_path = PROJ_PATH + "data/models/exp_1.7/rt_ad_1.joblib"
    sub_save_path = PROJ_PATH + "data/processed/1_7_subm_ad_1.csv"
    test_df_path = PROJ_PATH + "data/raw/test_data.csv"

    needest_columns = [
        'DIP', 'AVG_ORIG_OIL_SAT', 'ORIG_OIL_H', 'TOTAL_INJ', 'TOTAL_GNTL_INJ',
        'Lin_Dist_Inj_Factor', 'SGMT_CUM_STM_INJ_1', 'FT_DIST_PAT_1', 'SGMT_CUM_STM_INJ_2',
        'FT_DIST_PAT_2', 'SGMT_CUM_STM_INJ_3', 'FT_DIST_PAT_3', 'TOTAL_PROD', 'Lin_Dist_Prod_Factor'
    ]
    group_col = 'CMPL_FAC_ID'
    time_col = 'SURV_DTE'

    df = pd.read_csv(test_df_path)
    preproc_df = preproc_1_7_remake(df, base_features=needest_columns + [group_col] + [time_col],
                                    add_norm_total_inj=True)#!!!!

    # get meedest columns
    needest_sand_columns = [c for c in preproc_df.columns if 'SAND_' in c]
    needest_columns = needest_columns + \
                      needest_sand_columns + \
                      ['steam_inj_nums'] + \
                      ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time'] + \
                      ['total_inj_fac', 'total_inj_fac_sand']   # normalized total_inj, total_gntl_inj features

    X = preproc_df[needest_columns]
    print(X.columns)
    X = X.to_numpy()

    # fit, scoring, save
    reg_trees = joblib.load(model_path)
    preds = reg_trees.predict(X)

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub.to_csv(sub_save_path, index=False)