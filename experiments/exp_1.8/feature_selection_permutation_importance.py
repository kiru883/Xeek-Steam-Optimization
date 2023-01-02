import pandas as pd
import numpy as np
import eli5
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
from sklearn.model_selection import GroupKFold
from eli5.sklearn import PermutationImportance
from modules.scoring.cross_val_score import get_cv_rmse_score_1
from modules.model.Kmeans_RF import Kmeans2c_RF, GMM_RF
from modules.model.GBcls_RFtree import GBcls_RFtree

if __name__ == '__main__':
    # model params
    random_state = 42
    CV = 4
    n_trees = 300#3000 #
    criterion = 'squared_error'

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
    time_col = 'SURV_DTE'

    df = pd.read_csv(train_df_path)

    preproc_df = preproc_1_7_remake(df, base_features=needest_columns + [target_col] + [group_col] + [time_col],
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


    # test model
    model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state, criterion=criterion, n_jobs=-1, max_features=1.0)


    X, y, group = preproc_df[needest_columns], preproc_df[target_col], preproc_df[group_col]
    print(f'Columns: {X.columns}')
    X, y, group = X.to_numpy(), y.to_numpy().reshape(-1, 1), group.to_numpy().reshape(-1, 1)

    gkf = GroupKFold(n_splits=4)
    perm_means, perm_stds = [], []
    for train_index, test_index in gkf.split(X, y, groups=group):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        print('Next fold...')

        mod = model.fit(X_train, y_train)

        perm = PermutationImportance(mod,
                                     scoring=RMSE,
                                     n_iter=200,
                                     random_state=random_state)
        _fitted_perm = perm.fit(X_test, y_test)

        perm_means.append(_fitted_perm.feature_importances_)
        perm_stds.append(_fitted_perm.feature_importances_std_)

    perm_means = np.array(perm_means)
    perm_stds = np.array(perm_stds)

    perm_means = np.mean(perm_means, axis=0)
    perm_stds = np.mean(perm_stds, axis=0)

    perm.feature_importances_ = perm_means
    perm.feature_importances_std_ = perm_stds

    w = eli5.show_weights(perm, feature_names=needest_columns, top=100)
    result = pd.read_html(w.data)[0]
    print(result)


