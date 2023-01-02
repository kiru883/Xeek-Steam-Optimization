import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer
from sklearn.cluster import *
from sklearn.linear_model import LinearRegression, Lasso, RANSACRegressor, SGDRegressor, PoissonRegressor, TweedieRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, GradientBoostingRegressor
from modules.processing.experiment_preprocessings import preproc_1_7_remake
from modules.scoring.metrics import RMSE
from sklearn.gaussian_process import GaussianProcessRegressor
from modules.scoring.cross_val_score import get_cv_rmse_score_1
from modules.model.Kmeans_RF import Kmeans2c_RF, GMM_RF
from modules.model.GBcls_RFtree import GBcls_RFtree


#wt lindist etc LB 0.3107

if __name__ == '__main__':
    # model params
    random_state = 42
    CV = 4
    n_trees = 3000 #
    criterion = 'squared_error'

    # data paths, params
    PROJ_PATH = "/home/kirill/projects/personal_projects/xeek-Steam-Optimization/"
    model_path = PROJ_PATH + "data/models/exp_1.8/rt_ad_stacking.joblib"
    sub_save_path = PROJ_PATH + "data/processed/1_8_subm_ad_10.csv"
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
                                    add_norm_total_inj=True,# 1.7.1
                                    add_norm_stm_inj=True,# 1.7.2
                                    add_sand_groups=True,# 1.7.3
                                    add_mean_sand_group=True, # 1.7.6
                                    add_groups_num=True) # 1.7.8

    # get meedest columns
    needest_sand_columns = [c for c in preproc_df.columns if 'SAND_' in c]
    needest_columns = needest_columns + \
                      needest_sand_columns + \
                      ['steam_inj_nums'] + \
                      ['seq_num_un', 'fcl_life_time', 'fcl_sand_life_time'] + \
                      ['total_inj_fac', 'total_inj_fac_sand', 'total_prod_fac'] + \
                      ['cum_stm1_fac_sand', 'cum_stm2_fac_sand', 'cum_stm3_fac_sand'] + \
                      ['mean_group_orig_sat', 'mean_group_sinj_gntl'] + \
                      ['groups_num']

    BAD_FEATURES = [
        'Lin_Dist_Prod_Factor', 'Lin_Dist_Inj_Factor'
    ]
    needest_columns = [i for i in needest_columns if i not in BAD_FEATURES]

    X = preproc_df[needest_columns]
    print(X.columns)
    X = X.to_numpy()

    # fit, scoring, save
    model = joblib.load(model_path)
    preds = model.predict(X)

    df_sub = pd.DataFrame({'PCT_DESAT_TO_ORIG': preds})
    df_sub.to_csv(sub_save_path, index=False)


